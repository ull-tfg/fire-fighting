import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QMixNet(nn.Module):
    """Red de mezcla para QMIX que combina los valores Q de cada agente en un valor Q global"""
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super(QMixNet, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        
        # Red para generar los parámetros de la primera capa de mezcla (weights)
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim)
        )
        
        # Red para generar los parámetros de la segunda capa de mezcla (weights)
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim * 1)
        )
        
        # Red para generar los parámetros de bias
        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
    def forward(self, agent_qs, states):
        """
        Combina los valores Q individuales en un valor Q global
        
        Args:
            agent_qs: Tensor de valores Q para cada agente [batch_size, num_agents]
            states: Tensor de estados globales [batch_size, state_dim]
            
        Returns:
            q_tot: Valor Q global [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        
        # Primera capa de mezcla
        w1 = self.hyper_w_1(states).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(batch_size, 1, self.embed_dim)
        
        # Aplicar activación - importante para mantener la monotonicidad
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        # Segunda capa de mezcla
        w2 = self.hyper_w_2(states).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b_2(states).view(batch_size, 1, 1)
        
        # Generar el valor Q global
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(batch_size, 1)

class QMixer:
    """Implementación del algoritmo QMIX"""
    def __init__(self, num_agents, state_dim, device, mixing_embed_dim=32, gamma=0.99, lr=0.001, tau=0.01):
        self.num_agents = num_agents
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Redes de mezcla (actual y objetivo)
        self.mixer = QMixNet(num_agents, state_dim, mixing_embed_dim).to(device)
        self.target_mixer = QMixNet(num_agents, state_dim, mixing_embed_dim).to(device)
        
        # Copiar weights iniciales al target mixer
        self.update_target_mixer(tau=1.0)
        
        # Optimizador
        self.optimizer = torch.optim.Adam(self.mixer.parameters(), lr=lr)
    
    def update_target_mixer(self, tau=None):
        """Actualiza la red objetivo con interpolación Polyak"""
        if tau is None:
            tau = self.tau
            
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def mix(self, agent_qs, states):
        """Mezcla los valores Q individuales a un valor global usando el estado del sistema"""
        return self.mixer(agent_qs, states)
    
    def target_mix(self, agent_qs, states):
        """Mezcla usando la red objetivo"""
        return self.target_mixer(agent_qs, states)