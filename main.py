import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from environment import FirefightingEnv
from agent import DQNAgent
from exact_graph import generate_exact_graph, visualize_exact_graph

EPISODES = 1500
rewards_history = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph = generate_exact_graph()
visualize_exact_graph()

env = FirefightingEnv(num_agents = 3, graph=graph)

agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=np.prod(env.action_space.nvec), action_space=env.action_space)

obs_dim = env.observation_space.shape[0]
action_dim = np.prod(env.action_space.nvec)

# Agrega al inicio:
steps_per_episode = []
# Dentro del bucle de episodios:
for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        # env.render()  # visual con pygame
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1
        agent.train_step()
        agent.soft_update_target_network()

    rewards_history.append(total_reward)
    steps_per_episode.append(steps)


    agent.decay_epsilon()

    if ep % 10 == 0:
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")




# Después del entrenamiento:
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(rewards_history)
plt.title("Recompensa total por episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")

plt.subplot(1, 3, 2)
plt.plot(steps_per_episode)
plt.title("Número de pasos por episodio")
plt.xlabel("Episodio")
plt.ylabel("Pasos")


plt.subplot(1, 3, 3)
plt.plot(agent.losses)
plt.title("Pérdida (Loss) durante el entrenamiento")
plt.xlabel("Paso de entrenamiento")
plt.ylabel("Loss")
plt.grid()
plt.show()
