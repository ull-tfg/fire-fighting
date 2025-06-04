import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import json
from datetime import datetime

from environment import FirefightingEnv
from agent import DQNAgent
from graph_generator import generate_variable_graphs, save_graphs_to_files
from pygame_viz import visualize_trained_agent_interactive

# Crear la carpeta de resultados si no existe
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Generar un timestamp para este experimento
timestamp = time.strftime("%Y%m%d-%H%M%S")
experiment_dir = os.path.join(results_dir, f"multi_graph_experiment_{timestamp}")
os.makedirs(experiment_dir)

# Configuración del experimento
EPISODES_PER_GRAPH = 2000
MAX_STEPS = 500
NUM_GRAPHS = 5
EVAL_INTERVAL = 100  # Evaluar cada 100 episodios

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def evaluate_agent(agent, env, num_eval_episodes=5):
    """
    Evalúa el rendimiento del agente sin exploración.
    
    Args:
        agent: El agente DQN
        env: El entorno
        num_eval_episodes: Número de episodios de evaluación
    
    Returns:
        Diccionario con métricas de evaluación
    """
    eval_rewards = []
    eval_fires = []
    eval_steps = []
    
    # Guardar epsilon original y establecerlo a 0 para evaluación
    original_eps = agent.epsilon
    agent.epsilon = 0.0  # Sin exploración durante evaluación
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        initial_fires = sum(level > 0 for level in env.fire_levels.values())
        steps_done = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps_done += 1
            if steps_done > MAX_STEPS:
                break
        
        # Calcular fuegos extinguidos
        remaining_fires = sum(level > 0 for level in env.fire_levels.values())
        fires_extinguished = initial_fires - remaining_fires
        
        eval_rewards.append(total_reward)
        eval_fires.append(fires_extinguished)
        eval_steps.append(steps_done)
    
    # Restaurar epsilon original
    agent.epsilon = original_eps
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'mean_fires': np.mean(eval_fires),
        'mean_steps': np.mean(eval_steps),
        'all_rewards': eval_rewards,
        'all_fires': eval_fires
    }

def visualize_single_graph_performance(metrics, graph_id, save_dir):
    """
    Visualiza el rendimiento para un solo grafo.
    """
    plt.figure(figsize=(16, 12))
    
    # Recompensa máxima teórica
    MAX_THEORETICAL_REWARD = 227.5
    
    # Plot team rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.axhline(y=MAX_THEORETICAL_REWARD, color='green', linestyle='--', label='Max Theoretical Reward')
    plt.title(f'Graph {graph_id} - Team Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot smoothed team rewards
    plt.subplot(3, 2, 2)
    window_size = min(50, len(metrics['episode_rewards']))
    if window_size > 1:
        smoothed_rewards = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards)
        plt.axhline(y=MAX_THEORETICAL_REWARD, color='green', linestyle='--', label='Max Theoretical Reward')
        plt.title(f'Graph {graph_id} - Smoothed Team Rewards (Window={window_size})')
    else:
        plt.title(f'Graph {graph_id} - Smoothed Team Rewards (Need more episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # Plot fires extinguished
    plt.subplot(3, 2, 3)
    plt.plot(metrics['fires_extinguished'])
    plt.title(f'Graph {graph_id} - Fires Extinguished per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Fires Extinguished')
    
    # Plot steps per episode
    plt.subplot(3, 2, 4)
    plt.plot(metrics['steps_per_episode'])
    plt.title(f'Graph {graph_id} - Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot training loss
    plt.subplot(3, 2, 5) 
    if 'losses' in metrics and len(metrics['losses']) > 0:
        plt.plot(metrics['losses'])
        plt.title(f'Graph {graph_id} - Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
    else:
        plt.title(f'Graph {graph_id} - Training Loss (No data)')
    
    # Plot evaluation metrics
    plt.subplot(3, 2, 6)
    if len(metrics['eval_rewards']) > 0:
        eval_episodes = np.arange(0, len(metrics['episode_rewards']), EVAL_INTERVAL)[:len(metrics['eval_rewards'])]
        plt.plot(eval_episodes, metrics['eval_rewards'], 'ro-', label='Evaluation Rewards')
        plt.title(f'Graph {graph_id} - Evaluation Performance')
        plt.xlabel('Episode')
        plt.ylabel('Average Evaluation Reward')
        plt.legend()
    else:
        plt.title(f'Graph {graph_id} - Evaluation Performance (No data)')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'graph_{graph_id}_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico del Grafo {graph_id} guardado en: {plot_path}")

def visualize_multi_graph_comparison(all_results, save_dir):
    """
    Visualiza la comparación entre todos los grafos.
    """
    plt.figure(figsize=(20, 15))
    
    # 1. Comparación de recompensas finales
    plt.subplot(3, 3, 1)
    final_rewards = []
    graph_labels = []
    for i, results in enumerate(all_results):
        if results['metrics']['episode_rewards']:
            final_rewards.append(np.mean(results['metrics']['episode_rewards'][-100:]))  # Últimos 100 episodios
            graph_labels.append(f"Graph {i+1}")
    
    plt.bar(graph_labels, final_rewards, alpha=0.7)
    plt.title('Final Performance Comparison (Last 100 Episodes)')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    
    # 2. Curvas de aprendizaje superpuestas
    plt.subplot(3, 3, 2)
    for i, results in enumerate(all_results):
        rewards = results['metrics']['episode_rewards']
        if len(rewards) > 50:
            # Suavizar las curvas
            window_size = 50
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed, label=f'Graph {i+1}', alpha=0.8)
    plt.title('Learning Curves Comparison (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # 3. Comparación de fuegos extinguidos
    plt.subplot(3, 3, 3)
    final_fires = []
    for i, results in enumerate(all_results):
        if results['metrics']['fires_extinguished']:
            final_fires.append(np.mean(results['metrics']['fires_extinguished'][-100:]))
    
    plt.bar(graph_labels, final_fires, alpha=0.7, color='orange')
    plt.title('Fires Extinguished Comparison (Last 100 Episodes)')
    plt.ylabel('Average Fires Extinguished')
    plt.xticks(rotation=45)
    
    # 4. Eficiencia en pasos
    plt.subplot(3, 3, 4)
    final_steps = []
    for i, results in enumerate(all_results):
        if results['metrics']['steps_per_episode']:
            final_steps.append(np.mean(results['metrics']['steps_per_episode'][-100:]))
    
    plt.bar(graph_labels, final_steps, alpha=0.7, color='green')
    plt.title('Steps per Episode Comparison (Last 100 Episodes)')
    plt.ylabel('Average Steps')
    plt.xticks(rotation=45)
    
    # 5. Convergencia (varianza en últimos episodios)
    plt.subplot(3, 3, 5)
    convergence_scores = []
    for i, results in enumerate(all_results):
        if len(results['metrics']['episode_rewards']) >= 100:
            convergence_scores.append(np.std(results['metrics']['episode_rewards'][-100:]))
    
    plt.bar(graph_labels, convergence_scores, alpha=0.7, color='red')
    plt.title('Convergence Stability (Std of Last 100 Episodes)')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    
    # 6. Evaluación final de cada grafo
    plt.subplot(3, 3, 6)
    eval_rewards = []
    eval_stds = []
    for i, results in enumerate(all_results):
        if results['final_eval']['mean_reward'] is not None:
            eval_rewards.append(results['final_eval']['mean_reward'])
            eval_stds.append(results['final_eval']['std_reward'])
    
    plt.bar(graph_labels, eval_rewards, yerr=eval_stds, alpha=0.7, color='purple', capsize=5)
    plt.title('Final Evaluation Performance')
    plt.ylabel('Average Evaluation Reward')
    plt.xticks(rotation=45)
    
    # 7. Características de los grafos
    plt.subplot(3, 3, 7)
    graph_sizes = []
    for i, results in enumerate(all_results):
        graph_sizes.append(results['graph_info']['num_nodes'])
    
    plt.bar(graph_labels, graph_sizes, alpha=0.7, color='cyan')
    plt.title('Graph Sizes (Number of Nodes)')
    plt.ylabel('Number of Nodes')
    plt.xticks(rotation=45)
    
    # 8. Complejidad de los grafos (aristas)
    plt.subplot(3, 3, 8)
    graph_edges = []
    for i, results in enumerate(all_results):
        graph_edges.append(results['graph_info']['num_edges'])
    
    plt.bar(graph_labels, graph_edges, alpha=0.7, color='yellow')
    plt.title('Graph Complexity (Number of Edges)')
    plt.ylabel('Number of Edges')
    plt.xticks(rotation=45)
    
    # 9. Densidad de incendios
    plt.subplot(3, 3, 9)
    fire_densities = []
    for i, results in enumerate(all_results):
        fire_densities.append(results['graph_info']['num_fires'])
    
    plt.bar(graph_labels, fire_densities, alpha=0.7, color='red')
    plt.title('Fire Density (Number of Fires)')
    plt.ylabel('Number of Fires')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'multi_graph_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparación multi-grafo guardada en: {comparison_path}")

# Función record_episode removida - no necesaria para este experimento

def train_on_single_graph(graph, graph_id, save_dir):
    """
    Entrena al agente en un solo grafo.
    
    Returns:
        Diccionario con resultados del entrenamiento
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO EN GRAFO {graph_id}")
    print(f"{'='*60}")
    
    # Crear entorno con el grafo
    env = FirefightingEnv(num_agents=3, graph=graph)
    
    # Crear nuevo agente para cada grafo
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0], 
        action_dim=np.prod(env.action_space.nvec), 
        action_space=env.action_space
    )
    
    # Métricas de entrenamiento
    rewards_history = []
    steps_per_episode = []
    fires_extinguished_history = []
    eval_rewards_history = []
    
    # Información del grafo
    graph_info = {
        'num_nodes': len(graph.nodes()),
        'num_edges': len(graph.edges()),
        'num_depots': len([n for n, d in graph.nodes(data=True) if d.get('type') == 'depot']),
        'num_fires': len([n for n, d in graph.nodes(data=True) if d.get('type') == 'fire']),
        'num_starters': len([n for n, d in graph.nodes(data=True) if d.get('type') == 'starter'])
    }
    
    print(f"Características del grafo:")
    print(f"  - Nodos: {graph_info['num_nodes']}")
    print(f"  - Aristas: {graph_info['num_edges']}")
    print(f"  - Depósitos: {graph_info['num_depots']}")
    print(f"  - Incendios: {graph_info['num_fires']}")
    print(f"  - Starters: {graph_info['num_starters']}")
    
    # Bucle de entrenamiento
    for ep in range(EPISODES_PER_GRAPH):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Registrar fuegos iniciales
        initial_fires = sum(level > 0 for level in env.fire_levels.values())
        
        while not done and steps < MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Almacenar experiencia
            agent.store_transition(state, action, reward, next_state, done)
            
            # Entrenar el agente
            agent.train_step()
            
            # Actualizar red target
            agent.soft_update_target_network()
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Calcular fuegos extinguidos
        remaining_fires = sum(level > 0 for level in env.fire_levels.values())
        fires_extinguished = initial_fires - remaining_fires
        
        # Almacenar métricas
        rewards_history.append(total_reward)
        steps_per_episode.append(steps)
        fires_extinguished_history.append(fires_extinguished)
        
        # Decaer epsilon
        agent.decay_epsilon()
        
        # Evaluación periódica
        if ep % EVAL_INTERVAL == 0:
            eval_result = evaluate_agent(agent, env, num_eval_episodes=5)
            eval_rewards_history.append(eval_result['mean_reward'])
            print(f"Episodio {ep}: Reward={total_reward:.2f}, Eval_Reward={eval_result['mean_reward']:.2f}, "
                  f"Fires={fires_extinguished}, Steps={steps}, Epsilon={agent.epsilon:.3f}")
        

    
    # Evaluación final
    print(f"\nEvaluación final del Grafo {graph_id}...")
    final_eval = evaluate_agent(agent, env, num_eval_episodes=20)
    
    # Compilar métricas
    metrics = {
        'episode_rewards': rewards_history,
        'steps_per_episode': steps_per_episode,
        'fires_extinguished': fires_extinguished_history,
        'losses': agent.get_losses(),
        'eval_rewards': eval_rewards_history
    }
    
    # Crear visualización para este grafo
    visualize_single_graph_performance(metrics, graph_id, save_dir)
    
    print(f"Grafo {graph_id} completado:")
    print(f"  - Reward promedio últimos 100 episodios: {np.mean(rewards_history[-100:]):.2f}")
    print(f"  - Evaluación final: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  - Fuegos promedio extinguidos: {np.mean(fires_extinguished_history[-100:]):.2f}")
    
    return {
        'metrics': metrics,
        'final_eval': final_eval,
        'graph_info': graph_info,
        'agent': agent  # Guardamos el agente entrenado
    }

def save_experiment_summary(all_results, save_dir):
    """
    Guarda un resumen del experimento en formato JSON y CSV.
    """
    # Resumen en formato JSON
    summary = {
        'experiment_config': {
            'episodes_per_graph': EPISODES_PER_GRAPH,
            'max_steps': MAX_STEPS,
            'num_graphs': NUM_GRAPHS,
            'eval_interval': EVAL_INTERVAL,
            'device': str(device)
        },
        'results_summary': []
    }
    
    # Crear DataFrame para CSV
    csv_data = []
    
    for i, results in enumerate(all_results):
        graph_summary = {
            'graph_id': i + 1,
            'num_nodes': results['graph_info']['num_nodes'],
            'num_edges': results['graph_info']['num_edges'],
            'num_depots': results['graph_info']['num_depots'],
            'num_fires': results['graph_info']['num_fires'],
            'num_starters': results['graph_info']['num_starters'],
            'final_eval_reward': results['final_eval']['mean_reward'],
            'final_eval_std': results['final_eval']['std_reward'],
            'avg_reward_last_100': np.mean(results['metrics']['episode_rewards'][-100:]) if results['metrics']['episode_rewards'] else 0,
            'avg_fires_last_100': np.mean(results['metrics']['fires_extinguished'][-100:]) if results['metrics']['fires_extinguished'] else 0,
            'avg_steps_last_100': np.mean(results['metrics']['steps_per_episode'][-100:]) if results['metrics']['steps_per_episode'] else 0,
            'convergence_std': np.std(results['metrics']['episode_rewards'][-100:]) if len(results['metrics']['episode_rewards']) >= 100 else 0
        }
        summary['results_summary'].append(graph_summary)
        csv_data.append(graph_summary)
    
    # Guardar JSON
    json_path = os.path.join(save_dir, 'experiment_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Guardar CSV
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(save_dir, 'experiment_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Resumen del experimento guardado en:")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")

def main():
    """
    Función principal del experimento multi-grafo.
    """
    print("="*80)
    print("EXPERIMENTO DE ENTRENAMIENTO MULTI-GRAFO")
    print("="*80)
    print(f"Configuración:")
    print(f"  - Número de grafos: {NUM_GRAPHS}")
    print(f"  - Episodios por grafo: {EPISODES_PER_GRAPH}")
    print(f"  - Máximo de pasos por episodio: {MAX_STEPS}")
    print(f"  - Dispositivo: {device}")
    print(f"  - Directorio de resultados: {experiment_dir}")
    print("="*80)
    
    # Generar grafos con LA MISMA configuración
    print("\nGenerando 5 grafos con la misma configuración...")
    
    # Configuración única para todos los grafos
    graph_config = {
        "connectivity": "low", 
        "num_depots": "medium", 
        "num_fires": "medium", 
        "edge_widths": "medium", 
        "transit_times": "medium"
    }
    
    print(f"Configuración aplicada a todos los grafos: {graph_config}")
    
    all_graphs = []
    for i in range(NUM_GRAPHS):
        print(f"Generando grafo {i+1} con configuración: {graph_config}")
        graphs = generate_variable_graphs(
            num_graphs=1,
            base_nodes=15,
            seed=42 + i,  # Semilla diferente para cada grafo para generar estructuras distintas
            **graph_config
        )
        all_graphs.extend(graphs)
    
    # Guardar los grafos generados
    save_graphs_to_files(all_graphs, base_filename="training_graph", save_images=True)
    
    # Entrenar en cada grafo
    all_results = []
    
    for i, graph in enumerate(all_graphs):
        graph_id = i + 1
        results = train_on_single_graph(graph, graph_id, experiment_dir)
        all_results.append(results)
        
        # Guardar resultados intermedios
        torch.save(results['agent'].policy_net.state_dict(), 
                  os.path.join(experiment_dir, f'graph_{graph_id}_model.pth'))
    
    # Crear visualización comparativa
    print("\nCreando visualización comparativa...")
    visualize_multi_graph_comparison(all_results, experiment_dir)
    
    # Guardar resumen del experimento
    save_experiment_summary(all_results, experiment_dir)
    
    print("\n" + "="*80)
    print("EXPERIMENTO COMPLETADO")
    print("="*80)
    print(f"Resultados guardados en: {experiment_dir}")
    print("\nResumen de resultados:")
    for i, results in enumerate(all_results):
        print(f"Grafo {i+1}: Reward final = {results['final_eval']['mean_reward']:.2f} ± {results['final_eval']['std_reward']:.2f}")
    
    # Encontrar el mejor y peor grafo
    best_graph_idx = max(range(len(all_results)), 
                        key=lambda i: all_results[i]['final_eval']['mean_reward'])
    worst_graph_idx = min(range(len(all_results)), 
                         key=lambda i: all_results[i]['final_eval']['mean_reward'])
    
    print(f"\nMejor rendimiento: Grafo {best_graph_idx + 1}")
    print(f"Peor rendimiento: Grafo {worst_graph_idx + 1}")
    
    # Calcular estadísticas de variabilidad
    final_rewards = [results['final_eval']['mean_reward'] for results in all_results]
    print(f"\nEstadísticas de variabilidad entre grafos:")
    print(f"  - Reward promedio: {np.mean(final_rewards):.2f}")
    print(f"  - Desviación estándar: {np.std(final_rewards):.2f}")
    print(f"  - Rango: {np.max(final_rewards) - np.min(final_rewards):.2f}")
    print(f"  - Coeficiente de variación: {(np.std(final_rewards)/np.mean(final_rewards)*100):.2f}%")
    
    # Opcional: Visualización interactiva del mejor agente
    print(f"\n¿Deseas ver la visualización interactiva del mejor agente? (Grafo {best_graph_idx + 1})")
    visualize_trained_agent_interactive(all_results[best_graph_idx]['agent'], 
                                      FirefightingEnv(num_agents=3, graph=all_graphs[best_graph_idx]), 
                                      max_steps=100)

if __name__ == "__main__":
    main()