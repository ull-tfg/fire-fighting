import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=None, activation_fn=F.relu):
        """
        Inicializa la red DQN.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de la acción
            hidden_layers: Lista con el número de neuronas en cada capa oculta
            activation_fn: Función de activación a usar
        """
        super(DQN, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128]  # Arquitectura por defecto
        
        self.activation_fn = activation_fn
        
        # Construir las capas dinámicamente
        layers = []
        
        # Primera capa: de state_dim a primera capa oculta
        layers.append(nn.Linear(state_dim, hidden_layers[0]))
        
        # Capas intermedias
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Capa de salida
        layers.append(nn.Linear(hidden_layers[-1], action_dim))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        """Propagación hacia adelante."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Aplicar activación a todas menos la última capa
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
        return x