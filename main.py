# Descripción: Script principal para entrenar a los agentes en el entorno de lucha contra incendios.
from graph_utils import *
from environment import *
from dqn import *
from agent import *
import time
import numpy as np

# Definir los tipos de vehículos
vehicle_types = {
    'small': {'max_water_capacity': 50, 'water_dispense_rate': 5, 'water_refill_rate': 10, 'width': 1},
    'medium': {'max_water_capacity': 100, 'water_dispense_rate': 10, 'water_refill_rate': 20, 'width': 2},
    'large': {'max_water_capacity': 200, 'water_dispense_rate': 20, 'water_refill_rate': 40, 'width': 3}
}

# Acciones posibles para los agentes
ACTIONS = {
    0: 'move',
    1: 'extinguish',
    2: 'refill'
}

# Configuración del entorno
Gr = generate_graph()
num_agents = 3
env = MultiAgentFirefightingEnv(Gr, num_agents, vehicle_types)

# Inicializar agentes
agents = []
for _ in range(num_agents):
    agent = DQNAgent(
        state_dim=5 * num_agents,
        action_dim= len(ACTIONS),
        vehicle_types=vehicle_types,
        vehicle_type=random.choice(list(vehicle_types.keys()))
    )
    agents.append(agent)

print(f"Using device: {agents[0].device}")

# Variables para seguimiento de métricas
start_time = time.time()
episode_rewards = []
fires_extinguished_per_episode = []
steps_per_episode = []

# Configuración del entrenamiento
num_episodes = 10000
max_steps_per_episode = 300
print("\nIniciando entrenamiento...")
print(f"{'Episodio':^10} | {'Reward':^10} | {'Fuegos':^8} | {'Pasos':^8} | {'Epsilon':^8} | {'Tiempo':^10} | {'ETA':^10}")
print("-" * 75)

# Ciclo de entrenamiento
for episode in range(num_episodes):
    episode_start = time.time()
    state = env.reset()
    
    # Contar fuegos iniciales
    initial_fires = sum(1 for node in env.graph.nodes 
                       if env.graph.nodes[node]['tipo'] == 'incendio')
    
    total_reward = 0
    steps = 0
    done = False
    # Ciclo de pasos dentro del episodio
    while not done and steps < max_steps_per_episode:
        actions = {}
        for i, agent in enumerate(agents):
            actions[i] = agent.act(state)
        
        next_state, rewards, done = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward
        
        # Entrenamiento de agentes
        for i, agent in enumerate(agents):
            agent.remember(state, actions[i], rewards[i], next_state, done)
            agent.replay()
        
        state = next_state
        steps += 1
    
    # Calcular métricas del episodio
    remaining_fires = sum(1 for node in env.graph.nodes 
                         if env.graph.nodes[node]['tipo'] == 'incendio' and env.graph.nodes[node]['water_to_extinguish'] > 0)
    fires_extinguished = initial_fires - remaining_fires
    
    # Guardar métricas
    episode_rewards.append(total_reward)
    fires_extinguished_per_episode.append(fires_extinguished)
    steps_per_episode.append(steps)
    
    # Calcular promedios móviles
    window = 10
    avg_reward = np.mean(episode_rewards[-window:]) if len(episode_rewards) >= window else np.mean(episode_rewards)
    avg_fires = np.mean(fires_extinguished_per_episode[-window:]) if len(fires_extinguished_per_episode) >= window else np.mean(fires_extinguished_per_episode)
    
    # Calcular tiempos
    elapsed_time = time.time() - start_time
    avg_time_per_episode = elapsed_time / (episode + 1)
    eta = avg_time_per_episode * (num_episodes - episode - 1)
    
    # Mostrar progreso
    print(f"{episode:^10d} | {avg_reward:^10.2f} | {avg_fires:^8.1f} | {steps:^8d} | {agents[0].epsilon:^8.2f} | {format_time(elapsed_time):^10s} | {format_time(eta):^10s}", end='\r')
    
    # Resumen cada 10 episodios
    if (episode + 1) % 10 == 0:
        print(f"\n{'=' * 75}")
        print(f"Resumen después de {episode + 1} episodios:")
        print(f"Reward promedio últimos 10: {np.mean(episode_rewards[-10:]):.2f}")
        print(f"Fuegos apagados promedio: {np.mean(fires_extinguished_per_episode[-10:]):.1f}")
        print(f"Tiempo transcurrido: {format_time(elapsed_time)}")
        print(f"{'=' * 75}\n")


# Resultados finales
print_results(start_time, episode_rewards, fires_extinguished_per_episode, steps_per_episode)
