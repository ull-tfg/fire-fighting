import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

from dqn import DQN

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Ventaja y valor streams
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        
        # Combinar valor y ventaja
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DQNAgent:
    def __init__(self, state_dim, action_dim, vehicle_types, vehicle_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vehicle_types = vehicle_types
        self.vehicle_type = vehicle_type

        # Hiperparámetros mejorados
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.gamma = 0.95
        self.batch_size = 128
        self.target_update_freq = 10

        # Double DQN con arquitectura Dueling
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.update_target_model()

        # Optimizador con gradiente clipping
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Memoria de experiencia con priorización
        self.memory = deque(maxlen=100000)
        
        # Contador de pasos para actualización de red objetivo
        self.training_steps = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def normalize_state(self, state):
        # Normalizar el estado para mejor estabilidad
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def act(self, state):
        state = self.normalize_state(state)
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)  # Solo 3 acciones posibles: 0, 1, 2
    
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            
            # Añadir ruido pequeño para romper empates
            noise = torch.randn_like(q_values) * 0.01
            return torch.argmax(q_values + noise).item()

    def remember(self, state, action, reward, next_state, done):
        # Normalizar estados
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        
        # Clip reward para estabilidad
        reward = np.clip(reward, -1, 1)
        
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Muestreo con prioridad simple
        indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in indices]

        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([x[4] for x in batch])).to(self.device)

        # Double DQN
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)

        # Huber loss para mejor estabilidad
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()

        # Actualizar epsilon con decay más lento al inicio
        if self.training_steps < 1000:
            decay = 0.9999
        else:
            decay = self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon * decay)

        # Actualizar modelo objetivo periódicamente
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_model()

        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)