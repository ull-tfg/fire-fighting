import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from dqn import CentralizedDQN


# Hyperparámetros
GAMMA = 0.98
LR = 1e-5
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
REPLAY_SIZE = 50000
BATCH_SIZE = 256
TAU = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_dim, action_dim, action_space=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.policy_net = CentralizedDQN(state_dim, action_dim).to(device)
        self.target_net = CentralizedDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR, weight_decay=1e-4)
        self.memory = deque(maxlen=REPLAY_SIZE)
        self.steps_done = 0
        self.epsilon = EPSILON_START
        self.tau = TAU
        self.losses = []
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            # Random action based on action space
            if self.action_space is not None:
                return np.array([random.randint(0, n - 1) for n in self.action_space.nvec])
            else:
                return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
            
            # Handle multi-discrete action space if provided
            if self.action_space is not None:
                action_indices = np.unravel_index(np.argmax(q_values), self.action_space.nvec)
                return np.array(action_indices)
            else:
                return np.argmax(q_values)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Fix: Convert to numpy arrays first
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

        # Handle action indices properly for multi-discrete action space
        if self.action_space is not None:
            action_indices = [np.ravel_multi_index(a, self.action_space.nvec) for a in actions]
        else:
            action_indices = actions

        # Fix: Also convert action_indices to numpy array
        action_tensor = torch.tensor(np.array(action_indices)).to(device)

        q_values = self.policy_net(states).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * GAMMA * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.steps_done += 1

        # Log loss occasionally
        if self.steps_done % 50 == 0:
            print(f"[Paso {self.steps_done}] Pérdida (loss): {loss.item():.4f}")
    
    def soft_update_target_network(self):
        """
        Actualización suave de la red target usando el parámetro tau.
        target_params = tau * policy_params + (1 - tau) * target_params
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def get_losses(self):
        return self.losses
    