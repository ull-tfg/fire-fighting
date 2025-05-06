import optuna
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from environment import FirefightingEnv
from agent import DQNAgent
from exact_graph import generate_exact_graph
import agent as agent_module

# Número de episodes para evaluar cada configuración
NUM_EVALUATION_EPISODES = 1000
# Número de trials de Optuna
N_TRIALS = 100

def objective(trial):
    # Hyperparámetros a optimizar
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.99, 0.999)
    batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
    tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    
    # Configurar el entorno
    graph = generate_exact_graph()
    env = FirefightingEnv(num_agents=3, graph=graph)
    
    # Configurar el agente con los hyperparámetros sugeridos
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.nvec.prod()
    
    # Sobrescribir temporalmente los valores globales en el módulo agent
    original_gamma = agent_module.GAMMA
    original_lr = agent_module.LR
    original_epsilon_decay = agent_module.EPSILON_DECAY
    original_batch_size = agent_module.BATCH_SIZE
    original_tau = agent_module.TAU
    
    agent_module.GAMMA = gamma
    agent_module.LR = lr
    agent_module.EPSILON_DECAY = epsilon_decay
    agent_module.BATCH_SIZE = batch_size
    agent_module.TAU = tau
    
    # Crear el agente
    agent = DQNAgent(state_dim, action_dim, env.action_space)
    
    # Variables para seguimiento
    rewards = []
    episode_lengths = []
    
    # Entrenamiento con los hyperparámetros actuales
    for episode in range(NUM_EVALUATION_EPISODES):
        # Fix: Handle env.reset() correctly based on its return value
        reset_result = env.reset()
        # Check if reset returns a tuple or just the state
        if isinstance(reset_result, tuple):
            state = reset_result[0]  # Extract state from tuple
        else:
            state = reset_result  # Reset returns just the state
            
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action = agent.select_action(state)
            step_result = env.step(action)
            
            # Handle different step() return formats
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:  # Old API or custom format
                next_state, reward, done, _ = step_result[:4]
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            agent.soft_update_target_network()
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if episode_steps > 500:  # Limitar longitud del episodio
                done = True
        
        agent.decay_epsilon()
        rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Reportar progreso intermedio
        if episode % 5 == 0:
            trial.report(np.mean(rewards[-5:]), episode)
            if trial.should_prune():
                # Restaurar los valores originales antes de terminar
                agent_module.GAMMA = original_gamma
                agent_module.LR = original_lr
                agent_module.EPSILON_DECAY = original_epsilon_decay
                agent_module.BATCH_SIZE = original_batch_size
                agent_module.TAU = original_tau
                raise optuna.exceptions.TrialPruned()
    
    # Restaurar los valores originales
    agent_module.GAMMA = original_gamma
    agent_module.LR = original_lr
    agent_module.EPSILON_DECAY = original_epsilon_decay
    agent_module.BATCH_SIZE = original_batch_size
    agent_module.TAU = original_tau
    
    # Retornar la media de las últimas 10 recompensas como métrica
    return np.mean(rewards[-10:])

def run_optimization():
    """Ejecuta el proceso de optimización y guarda los resultados"""
    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/optuna_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction="maximize",  # Maximize reward
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    print(f"Iniciando optimización con {N_TRIALS} trials...")
    
    # Ejecutar la optimización
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Optimización interrumpida por el usuario.")
    
    # Mostrar y guardar resultados
    print("\nMejores hiperparámetros encontrados:")
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Guardar hiperparámetros en JSON
    with open(f"{results_dir}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Mejores hiperparámetros guardados en {results_dir}/best_params.json")
    
    # Visualizar resultados
    try:
        # Historial de optimización
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/optimization_history.png")
        
        # Importancia de parámetros
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/param_importances.png")
        
        # Slice plot para cada parámetro
        for param in best_params:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_slice(study, params=[param])
            plt.tight_layout()
            plt.savefig(f"{results_dir}/slice_{param}.png")
        
        # Contour plot para pares de parámetros
        pairs = []
        params = list(best_params.keys())
        for i in range(len(params)):
            for j in range(i+1, len(params)):
                pairs.append((params[i], params[j]))
        
        for param1, param2 in pairs:
            plt.figure(figsize=(10, 8))
            optuna.visualization.matplotlib.plot_contour(study, params=[param1, param2])
            plt.tight_layout()
            plt.savefig(f"{results_dir}/contour_{param1}_{param2}.png")
        
        print(f"Gráficos guardados en {results_dir}")
    except Exception as e:
        print(f"Error al generar visualizaciones: {str(e)}")
    
    # Actualizar el archivo agent.py con los mejores hiperparámetros
    update_agent_hyperparams(best_params)
    
    return best_params

def update_agent_hyperparams(best_params):
    """Actualiza el archivo agent.py con los mejores hiperparámetros"""
    agent_file = "/root/Home/TFG/centralized/agent.py"
    
    try:
        with open(agent_file, 'r') as f:
            lines = f.readlines()
        
        # Buscar y actualizar las líneas de hiperparámetros
        for i, line in enumerate(lines):
            if line.startswith("GAMMA ="):
                lines[i] = f"GAMMA = {best_params['gamma']}\n"
            elif line.startswith("LR ="):
                lines[i] = f"LR = {best_params['lr']}\n"
            elif line.startswith("EPSILON_DECAY ="):
                lines[i] = f"EPSILON_DECAY = {best_params['epsilon_decay']}\n"
            elif line.startswith("BATCH_SIZE ="):
                lines[i] = f"BATCH_SIZE = {best_params['batch_size']}\n"
            elif line.startswith("TAU ="):
                lines[i] = f"TAU = {best_params['tau']}\n"
        
        # Guardar el archivo actualizado
        with open(agent_file, 'w') as f:
            f.writelines(lines)
        
        print(f"Archivo agent.py actualizado con los mejores hiperparámetros")
    except Exception as e:
        print(f"Error al actualizar agent.py: {str(e)}")

def plot_training_curve(study):
    """Visualiza la curva de entrenamiento del mejor trial"""
    best_trial = study.best_trial
    
    # Extraer valores de rewards por episodio
    values = []
    for step, value in best_trial.intermediate_values.items():
        values.append((step, value))
    
    values.sort(key=lambda x: x[0])
    episodes = [v[0] for v in values]
    rewards = [v[1] for v in values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Media (últimos 5 episodios)')
    plt.title('Curva de Aprendizaje - Mejor Trial')
    plt.grid(True)
    
    # Guarda el gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/learning_curve_{timestamp}.png")
    plt.close()

if __name__ == "__main__":
    # Ejecutar optimización
    best_params = run_optimization()
    
    print("\nResumen de la optimización:")
    print("==========================")
    print(f"Mejor valor obtenido: {best_params}")
    print("==========================")
    print("Optimización completada. Los mejores hiperparámetros han sido guardados.")