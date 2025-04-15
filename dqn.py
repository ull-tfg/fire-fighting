import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedDQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(CentralizedDQN, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
