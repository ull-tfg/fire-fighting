import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

from environment import FirefightingEnv
from agent import DQNAgent
from exact_graph import generate_exact_graph, visualize_exact_graph

# Crear la carpeta de resultados si no existe
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Generar un timestamp para este experimento
timestamp = time.strftime("%Y%m%d-%H%M%S")
experiment_dir = os.path.join(results_dir, f"experiment_{timestamp}")
os.makedirs(experiment_dir)

# Función de evaluación
def evaluate_agent(agent, env, num_eval_episodes=5):
    """
    Evalúa el rendimiento del agente sin exploración.
    
    Args:
        agent: El agente DQN
        env: El entorno
        num_eval_episodes: Número de episodios de evaluación
    
    Returns:
        Recompensa media sobre los episodios de evaluación
    """
    eval_rewards = []
    eval_fires = []
    
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
            # Usar política determinista (sin exploración)
            action = agent.select_action(state, in_transit_mask=env.agent_in_transit)
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
    
    # Restaurar epsilon original
    agent.epsilon = original_eps
    
    return np.mean(eval_rewards)

# Nueva visualización multi-agente
def visualize_multi_agent_performance(metrics, save_dir):
    """
    Visualize the multi-agent team's performance over time.
    
    Args:
        metrics: Dictionary of training metrics
    """
    plt.figure(figsize=(16, 12))
    
    # Recompensa máxima teórica
    MAX_THEORETICAL_REWARD = 227.5
    
    # Plot team rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_rewards'])
    # Añadir línea de recompensa máxima teórica
    plt.axhline(y=MAX_THEORETICAL_REWARD, color='green', linestyle='--', label='Max Theoretical Reward')
    plt.title('Team Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot smoothed team rewards
    plt.subplot(3, 2, 2)
    window_size = min(10, len(metrics['episode_rewards']))
    if window_size > 1:
        smoothed_rewards = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards)
        # Añadir línea de recompensa máxima teórica
        plt.axhline(y=MAX_THEORETICAL_REWARD, color='green', linestyle='--', label='Max Theoretical Reward')
        plt.title(f'Smoothed Team Rewards (Window={window_size})')
    else:
        plt.title(f'Smoothed Team Rewards (Need more episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # Plot fires extinguished
    plt.subplot(3, 2, 3)
    plt.plot(metrics['fires_extinguished'])
    plt.title('Fires Extinguished per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Fires Extinguished')
    
    # Plot steps per episode
    plt.subplot(3, 2, 4)
    plt.plot(metrics['steps_per_episode'])
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot training loss
    plt.subplot(3, 2, 5) 
    if 'losses' in metrics and len(metrics['losses']) > 0:
        plt.plot(metrics['losses'], label='Training Loss', color='red')
        plt.title('Loss During Training')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
    else:
        plt.title('Training Loss (No Data)')
    
    # Plot evaluation metrics
    plt.subplot(3, 2, 6)
    if len(metrics['eval_rewards']) > 0:
        eval_x = list(range(0, len(metrics['episode_rewards']), 10))
        eval_x = eval_x[:len(metrics['eval_rewards'])]
        
        plt.plot(eval_x, metrics['eval_rewards'], label='Eval Rewards', color='blue')
        # Añadir línea de recompensa máxima teórica
        plt.axhline(y=MAX_THEORETICAL_REWARD, color='green', linestyle='--', label='Max Theoretical Reward')
        plt.title('Evaluation Metrics')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
    else:
        plt.title('Evaluation Metrics (No Data)')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_performance.png')
    plt.savefig(plot_path)
    plt.show()

# Función para registrar acciones y estados
def record_episode(agent, env, episode_number, save_dir):
    """
    Registra las acciones y estados de un episodio completo.
    
    Args:
        agent: El agente DQN
        env: El entorno
        episode_number: Número del episodio
        save_dir: Directorio donde guardar los registros
    """
    state = env.reset()
    done = False
    step = 0
    
    # Inicializar registro de datos
    episode_data = {
        'steps': [],
        'actions': [],
        'rewards': [],
        'states': [],
        'agent_positions': [],
        'transit_status': [],
        'water_levels': [],
        'fire_levels': [],
        'final_destinations': [],      # Destinos finales de cada agente
        'transit_targets': []          # Siguiente destino en la ruta de cada agente
    }
    
    while not done:
        # Seleccionar acción
        action = agent.select_action(state, in_transit_mask=env.agent_in_transit)
        
        # Registrar estado y acción
        episode_data['steps'].append(step)
        episode_data['actions'].append(action.tolist() if hasattr(action, 'tolist') else action)
        episode_data['states'].append(state.tolist() if hasattr(state, 'tolist') else state)
        episode_data['agent_positions'].append([env.agent_positions[i] for i in range(env.num_agents)])
        episode_data['transit_status'].append([env.agent_in_transit[i] for i in range(env.num_agents)])
        episode_data['water_levels'].append([env.agent_water[i] for i in range(env.num_agents)])
        episode_data['fire_levels'].append(list(env.fire_levels.values()))
        
        # Registrar destinos finales y siguientes
        final_dests = []
        transit_targets = []
        for i in range(env.num_agents):
            final_dest = env.final_destinations[i] if env.final_destinations[i] is not None else "None"
            transit_target = env.agent_transit_target[i] if env.agent_transit_target[i] is not None else "None"
            final_dests.append(final_dest)
            transit_targets.append(transit_target)
        
        episode_data['final_destinations'].append(final_dests)
        episode_data['transit_targets'].append(transit_targets)
        
        # Ejecutar acción
        next_state, reward, done, info = env.step(action)
        episode_data['rewards'].append(reward)
        
        state = next_state
        step += 1
        
        if step > MAX_STEPS:
            break
    
    # Guardar datos del episodio
    df = pd.DataFrame(episode_data)
    
    # Crear subdirectorio para episodios
    episodes_dir = os.path.join(save_dir, 'episodes')
    if not os.path.exists(episodes_dir):
        os.makedirs(episodes_dir)
        
    # Guardar CSV
    filename = os.path.join(episodes_dir, f'episode_{episode_number}_data.csv')
    df.to_csv(filename, index=False)
    
    # También guardar un registro de texto más legible
    txt_filename = os.path.join(episodes_dir, f'episode_{episode_number}_readable.txt')
    
    with open(txt_filename, 'w') as f:
        f.write(f"=== EPISODIO {episode_number} ===\n\n")
        
        for s in range(step):
            f.write(f"Paso {s}:\n")
            f.write(f"  Posiciones: {episode_data['agent_positions'][s]}\n")
            f.write(f"  En tránsito: {episode_data['transit_status'][s]}\n")
            f.write(f"  Destino final: {episode_data['final_destinations'][s]}\n")
            f.write(f"  Próximo destino: {episode_data['transit_targets'][s]}\n")
            f.write(f"  Acción seleccionada: {episode_data['actions'][s]}\n")
            f.write(f"  Niveles de agua: {episode_data['water_levels'][s]}\n")
            f.write(f"  Niveles de fuego: {episode_data['fire_levels'][s]}\n")
            f.write(f"  Recompensa: {episode_data['rewards'][s]}\n\n")
    
    print(f"Datos del episodio {episode_number} guardados en {filename}")

EPISODES = 2000
MAX_STEPS = 500
rewards_history = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph = generate_exact_graph()
visualize_exact_graph()

env = FirefightingEnv(num_agents = 3, graph=graph)

agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=np.prod(env.action_space.nvec), action_space=env.action_space)

obs_dim = env.observation_space.shape[0]
action_dim = np.prod(env.action_space.nvec)

# Inicializar métricas adicionales para la visualización avanzada
steps_per_episode = []
fires_extinguished_history = []
agent_rewards_history = [[] for _ in range(env.num_agents)]
eval_rewards_history = []

# Dentro del bucle de episodios:
for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # Registrar fuegos iniciales para calcular los extinguidos
    initial_fires = sum(level > 0 for level in env.fire_levels.values())
    agent_rewards = [0] * env.num_agents  # Para registrar recompensas individuales
    
    while not done:
        # env.render()  # visual con pygame
        action = agent.select_action(state, in_transit_mask=env.agent_in_transit)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

        agent.train_step()
        agent.soft_update_target_network()

        if steps > MAX_STEPS:
            break

    # Calcular fuegos extinguidos
    remaining_fires = sum(level > 0 for level in env.fire_levels.values())
    fires_extinguished = initial_fires - remaining_fires
    
    # Almacenar métricas
    rewards_history.append(total_reward)
    steps_per_episode.append(steps)
    fires_extinguished_history.append(fires_extinguished)

    agent.decay_epsilon()

    # Evaluación periódica (sin exploración)
    if ep % 50 == 0:
        # Usar la función de evaluación
        eval_reward = evaluate_agent(agent, env)
        eval_rewards_history.append(eval_reward)
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Eval: {eval_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Guardar acciones y estados
    if ep == 0 or ep == EPISODES - 1:
        # Guardar solo el primer y último episodio
        record_episode(agent, env, ep, experiment_dir)

# Recopilar métricas para la visualización
metrics = {
    'episode_rewards': rewards_history,
    'steps_per_episode': steps_per_episode,
    'fires_extinguished': fires_extinguished_history,
    'losses': agent.losses,
    'eval_rewards': eval_rewards_history
}
# Mostrar visualización avanzada
visualize_multi_agent_performance(metrics, experiment_dir)