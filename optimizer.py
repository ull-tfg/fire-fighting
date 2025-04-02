import optuna
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from environment import FirefightingEnv
from agent import DQNAgent
from graph_utils import generate_graph
from main import train_multi_agent, evaluate_agents

graph = generate_graph()

def objective(trial):
    """Función objetivo para Optuna que entrena y evalúa agentes con hiperparámetros sugeridos."""
    # Hiperparámetros a optimizar
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    epsilon_decay = trial.suggest_int("epsilon_decay", 5000, 20000)

    # Hiperparámetros de red neuronal
    # Número de capas ocultas y neuronas por capa
    n_layers = trial.suggest_int("n_layers", 1, 3)  # Número de capas ocultas
    hidden_layers = []
    for i in range(n_layers):
        n_units = trial.suggest_categorical(f"n_units_l{i}", [64, 128, 256, 512])
        hidden_layers.append(n_units)
    
    # Optimizar función de activación
    activation_name = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"])
    if activation_name == "relu":
        activation_fn = F.relu
    elif activation_name == "leaky_relu":
        activation_fn = F.leaky_relu
    else:
        activation_fn = F.elu
    
    # Configurar el entorno
    steps = 100  # Maximum steps per episode
    num_agents = 3  # Number of firefighting agents
    vehicle_types = {
        0: {'capacity': 200, 'width': 2},  # Standard vehicle
        1: {'capacity': 100, 'width': 1},  # Large capacity, narrow spray
        2: {'capacity': 500, 'width': 4}   # Small capacity, wide spray
    }
    env = FirefightingEnv(graph=graph, max_steps=steps, num_agents=num_agents, 
                        vehicle_types=vehicle_types)
    # Crear agentes con los hiperparámetros sugeridos
    agents = []
    for agent_id in range(num_agents):
        state_dim = env.observation_space[agent_id].shape[0]
        action_dim = env.action_space[agent_id].n
        vehicle_type = f"agent_{agent_id}"

        agent = DQNAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            vehicle_type=vehicle_type,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn
        )
        # Configurar los hiperparámetros sugeridos
        agent.batch_size = batch_size
        agent.gamma = gamma
        agent.tau = tau
        agent.epsilon_decay = epsilon_decay
        agent.learning_rate = learning_rate
        agent.optimizer.param_groups[0]['lr'] = learning_rate
        agents.append(agent)
    
    # Entrenar el modelo con menos episodios para optimización rápida
    train_multi_agent(
        env=env, 
        agents=agents, 
        num_episodes=200,
        max_steps=steps,
        eval_freq=50,
        render_training=False
    )
    eval_reward, eval_fires = evaluate_agents(env, agents, num_eval_episodes=10)
    # Objetivo: maximizar la recompensa de evaluación
    return eval_reward

def optimize_hyperparameters(n_trials=50):
    """Ejecuta la optimización de hiperparámetros con Optuna."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    print(f"Mejor valor de recompensa: {study.best_value}")
    
    # Visualizar importancia de hiperparámetros
    param_importance = optuna.visualization.plot_param_importances(study)
    param_importance.write_image('hyperparameter_importance.png')
    
    # Visualizar la optimización
    optimization_history = optuna.visualization.plot_optimization_history(study)
    optimization_history.write_image('optimization_history.png')
    
    return study.best_params

if __name__ == "__main__":
    best_params = optimize_hyperparameters(n_trials=50)
    
    # Guardar los mejores hiperparámetros
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f)