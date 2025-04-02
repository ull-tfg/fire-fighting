import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

from dqn import DQN

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'global_state', 'next_global_state'))

# Buffer de experiencias para entrenamiento centralizado
class EpisodeBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, episode):
        self.buffer.append(episode)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class QMIXAgent:
    def __init__(self, state_dim, action_dim, vehicle_type, agent_id, hidden_layers=[128, 128], activation_fn=F.relu):
        # Configuración de dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Parámetros del entorno
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vehicle_type = vehicle_type
        self.agent_id = agent_id
        
        # Hiperparámetros
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.95
        self.epsilon_min = 0.1
        self.epsilon_decay = 5000
        self.tau = 0.005
        self.learning_rate = 0.0005
        
        # Inicializar redes
        self.policy_net = DQN(state_dim, action_dim, hidden_layers, activation_fn).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_layers, activation_fn).to(self.device)
        self.update_target_net(1.0)  # Copia inicial completa
        self.target_net.eval()
        
        # Inicializar optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Memoria de episodios actual
        self.episode_memory = []
        self.steps_done = 0
        
    def update_target_net(self, tau=None):
        """Actualiza la red objetivo con interpolación Polyak"""
        if tau is None:
            tau = self.tau
            
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def select_action(self, state, available_actions=None):
        """Selecciona una acción siguiendo una política epsilon-greedy"""
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
                        np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                
                # Aplicar máscara de acciones disponibles
                if available_actions is not None:
                    action_mask = torch.ones_like(q_values) * float('-inf')
                    for action_idx in available_actions:
                        action_mask[0, action_idx] = 0
                    masked_q_values = q_values + action_mask
                    return masked_q_values.max(1).indices.item()
                else:
                    return q_values.max(1).indices.item()
        else:
            # Exploración aleatoria entre acciones disponibles
            if available_actions is not None and len(available_actions) > 0:
                return random.choice(available_actions)
            else:
                return random.randrange(self.action_dim)
    
    def store_transition(self, state, action, reward, next_state, done, global_state=None, next_global_state=None):
        """Guarda una transición en la memoria de episodio actual"""
        self.episode_memory.append(
            Transition(state, action, reward, next_state, done, global_state, next_global_state)
        )
    
    def get_q_values(self, states):
        """Obtiene valores Q para un batch de estados"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        return self.policy_net(states_tensor)
    
    def get_target_q_values(self, states):
        """Obtiene valores Q objetivo para un batch de estados"""
        states_tensor = torch.FloatTensor(states).to(self.device)
        return self.target_net(states_tensor)