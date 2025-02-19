import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

from graph_utils import *

class MultiAgentFirefighterEnv(gym.Env):
    def __init__(self, num_agents=2, num_fires=10, num_water_sources=4):
        super(MultiAgentFirefighterEnv, self).__init__()
        
        self.num_agents = num_agents
        self.num_fires = num_fires
        self.num_water_sources = num_water_sources
        self.graph = generate_graph(num_fires, num_water_sources)
        
        # Estado de los agentes
        self.agent_positions = [None] * num_agents
        self.agent_water = [500] * num_agents
        self.max_water = 500
        
        self.max_connections = max(len(list(self.graph.neighbors(node))) for node in self.graph.nodes())
        self.action_space = spaces.MultiDiscrete([self.max_connections + 2] * num_agents)
        
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(num_agents, num_fires + num_water_sources + 1 + self.max_connections * 3),
            dtype=np.float32
        )
        
        self.fires_extinguished = 0
        self.max_steps = 200
        self.current_step = 0
        
    def reset(self):
        self.graph = generate_graph(self.num_fires, self.num_water_sources)
        self.agent_positions = [np.random.choice(list(self.graph.nodes())) for _ in range(self.num_agents)]
        self.agent_water = [self.max_water] * self.num_agents
        self.fires_extinguished = 0
        self.current_step = 0
        
        return self._get_observations(), {}
    
    def _get_observations(self):
        obs = []
        for i in range(self.num_agents):
            agent_obs = []
            
            position_encoding = np.zeros(self.num_fires + self.num_water_sources)
            node_index = int(self.agent_positions[i].split('_')[1])
            if 'incendio' in self.agent_positions[i]:
                position_encoding[node_index] = 1
            else:
                position_encoding[self.num_fires + node_index] = 1
            agent_obs.extend(position_encoding)
            
            agent_obs.append(self.agent_water[i] / self.max_water)
            
            neighbors = list(self.graph.neighbors(self.agent_positions[i]))
            neighbor_info = np.zeros(self.max_connections * 3)
            
            for j, neighbor in enumerate(neighbors):
                if j >= self.max_connections:
                    break
                neighbor_info[j * 3] = 1 if 'incendio' in neighbor else 0
                neighbor_info[j * 3 + 1] = self.graph.nodes[neighbor].get('water_to_extinguish', 0) / 100
                neighbor_info[j * 3 + 2] = self.graph[self.agent_positions[i]][neighbor]['tiempo_viaje'] / 5
            
            agent_obs.extend(neighbor_info)
            obs.append(agent_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, actions):
        self.current_step += 1
        rewards = [0] * self.num_agents
        dones = [False] * self.num_agents
        infos = [{}] * self.num_agents
        
        for i, action in enumerate(actions):
            neighbors = list(self.graph.neighbors(self.agent_positions[i]))
            
            if action < len(neighbors):
                next_node = neighbors[action]
                rewards[i] -= self.graph[self.agent_positions[i]][next_node]['tiempo_viaje']
                self.agent_positions[i] = next_node
            
            elif action == len(neighbors):
                if 'estanque' in self.agent_positions[i]:
                    water_needed = self.max_water - self.agent_water[i]
                    water_available = self.graph.nodes[self.agent_positions[i]]['water_capacity']
                    water_to_add = min(water_needed, water_available)
                    self.agent_water[i] += water_to_add
                    self.graph.nodes[self.agent_positions[i]]['water_capacity'] -= water_to_add
                    rewards[i] += water_to_add * 0.1
                else:
                    rewards[i] -= 5
            
            elif action == len(neighbors) + 1:
                if 'incendio' in self.agent_positions[i]:
                    fire_size = self.graph.nodes[self.agent_positions[i]]['water_to_extinguish']
                    if self.agent_water[i] >= fire_size and fire_size > 0:
                        self.agent_water[i] -= fire_size
                        self.graph.nodes[self.agent_positions[i]]['water_to_extinguish'] = 0
                        self.fires_extinguished += 1
                        rewards[i] += 100
                    else:
                        rewards[i] -= 10
                else:
                    rewards[i] -= 5
            
            if self.fires_extinguished == self.num_fires:
                rewards[i] += 500
                dones[i] = True
            elif self.current_step >= self.max_steps:
                dones[i] = True
            
        return self._get_observations(), rewards, dones, infos
    
    def render(self, mode='human'):
        if mode == 'human':
            visualize_graph(self.graph)
            plt.show()