import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

from dqn import CentralizedDQN


# Hyperparámetros
GAMMA = 0.9726280205180136
LR = 9.815084417451514e-06
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9986533647491379
REPLAY_SIZE = 50000
BATCH_SIZE = 384
TAU = 0.09790627055934469

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

        # Añadir registro de últimas acciones tomadas
        self.last_actions = np.zeros(shape=(action_space.nvec.shape[0],), dtype=int)
    
    def select_action(self, state, in_transit_mask=None):
        # Create a copy of the current action that we'll update
        selected_action = self.last_actions.copy()

        # Handle the case where in_transit_mask is a dictionary
        if isinstance(in_transit_mask, dict):
            # Create an empty boolean mask array
            transit_mask_array = np.zeros(self.action_space.nvec.shape[0], dtype=bool)

            # Fill in the True values for agents in transit
            for agent_id, is_in_transit in in_transit_mask.items():
                if is_in_transit:
                    transit_mask_array[agent_id] = True

            in_transit_mask = transit_mask_array
        elif in_transit_mask is None:
            in_transit_mask = np.zeros(self.action_space.nvec.shape[0], dtype=bool)

        # Only select new actions for agents that are NOT in transit
        agents_to_act = np.where(~np.array(in_transit_mask))[0]

        if len(agents_to_act) > 0:
            if random.random() < self.epsilon:
                # Random actions only for agents not in transit
                for agent_id in agents_to_act:
                    selected_action[agent_id] = random.randint(0, self.action_space.nvec[agent_id] - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = self.policy_net(state_tensor)

                    # Handle action selection for each agent not in transit
                    q_reshaped = q_values.view(*self.action_space.nvec)

                    for agent_id in agents_to_act:
                        # Get the best action for this specific agent
                        indices = [slice(None) if i == agent_id else selected_action[i] 
                                  for i in range(len(self.action_space.nvec))]
                        agent_q_values = q_reshaped[tuple(indices)]
                        selected_action[agent_id] = agent_q_values.argmax().item()

        # Update the last actions
        self.last_actions = selected_action.copy()
    
        return selected_action
    
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

        loss = F.huber_loss(q_values, target_q.detach())
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
