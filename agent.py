import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from enum import Enum

from replay_memory import *
from dqn import DQN

class Action(Enum):
    MOVE = 0
    REFILL = 1
    EXTINGUISH = 2
    

class DQNAgent:
    def __init__(self, state_dim, action_dim, vehicle_type):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vehicle_type = vehicle_type
        
        # Hyperparameters
        self.batch_size = 128 # is the number of samples to train on in a single batch
        self.gamma = 0.99  # is the discount factor for future rewards
        self.epsilon = 0.95
        self.epsilon_min = 0.1
        self.epsilon_decay = 10000  # Controls the rate of exponential decay for epsilon, higher is slower
        self.target_update_freq = 0.0005  # is the update rate of the target network
        self.learning_rate = 1e-4  # is the learning rate of the optimizer (AdamW)
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.update_target_model()
        self.target_net.eval()
        
        # Initialize replay memory with capacity
        self.memory = ReplayMemory(capacity=10000)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        
        self.steps_done = 0

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state = state.to(self.device)
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1).to(self.device)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
        
    def learn(self, batchsize):
        if len(self.memory) < batchsize:
            return
        
        # Sample batch from replay memory
        transitions = self.memory.sample(batchsize)
        batch = Transition(*zip(*transitions))
        states, actions, next_states, rewards = batch.state, batch.action, batch.next_state, batch.reward
        
        # Convertir a tensores
        # Convertir a tensores asegurando que están en el dispositivo correcto
        states = torch.tensor(np.array(batch.state), dtype=torch.float, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).reshape((-1, 1))
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float, device=self.device).reshape((-1, 1))
        dones = torch.tensor(batch.done, dtype=torch.float, device=self.device).reshape((-1, 1))
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        predicted_qs = self.policy_net(states).gather(1, actions).to(self.device)
        
        target_qs = self.target_net(next_states).max(1).values.reshape(-1, 1).to(self.device)
        
        y_js = rewards + self.gamma * target_qs
        
        loss = F.mse_loss(predicted_qs, y_js)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def load(self, name):
        self.policy_net.load_state_dict(torch.load(name))
        self.policy_net.eval()

    def save(self, name):
        torch.save(self.policy_net.state_dict(), name)