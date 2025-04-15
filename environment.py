import gym
from gym import spaces
import numpy as np
import networkx as nx
import random
# import pygame

class FirefightingEnv(gym.Env):
    def __init__(self, num_agents=3, graph=None, max_water=10, max_fire=50):
        super(FirefightingEnv, self).__init__()
        self.num_agents = num_agents
        self.max_water = max_water
        self.max_fire = max_fire

        #
        self.final_destinations = {}
        self.agent_transit_target = {}
        self.agent_transit_source = {}
        self.agent_in_transit = {}
        self.agent_transit_time_remaining = {}
        
        # Usar el grafo proporcionado en lugar de generar uno nuevo
        self.graph = graph
        
        # Extraer los nodos del grafo según su tipo
        self.fire_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                          if attrs.get('type') == 'fire']
        self.depot_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                           if attrs.get('type') == 'depot']
        self.starter_nodes = [node for node, attrs in self.graph.nodes(data=True) 
                             if attrs.get('type') == 'starter']
        
        # Todos los nodos para las acciones de los agentes
        self.nodes = self.fire_nodes + self.depot_nodes + self.starter_nodes
        
        # Inicializar posiciones iniciales de los agentes en los nodos de inicio
        if self.starter_nodes and len(self.starter_nodes) >= num_agents:
            self.initial_agents_positions = self.starter_nodes[:num_agents]
        else:
            self.initial_agents_positions = [random.choice(self.nodes) for _ in range(num_agents)]
            
        self.agent_positions = self.initial_agents_positions.copy()
        self.agent_water = [max_water for _ in range(num_agents)]
        
        # Inicializar niveles de fuego según los atributos del grafo
        self.fire_levels = {}
        for node in self.fire_nodes:
            water_to_extinguish = self.graph.nodes[node].get('water_to_extinguish', max_fire)
            self.fire_levels[node] = water_to_extinguish
        
        # Definir espacios de acción y observación
        self.action_space = spaces.MultiDiscrete([len(self.nodes)] * num_agents)
        # Actualizar obs_size para incluir todas las observaciones agregadas
        # Actualizar obs_size para incluir todas las observaciones agregadas
        obs_size = (
            num_agents +                          # posiciones
            num_agents +                          # agua de agentes
            len(self.fire_nodes) +                # fuego en nodos
            num_agents * len(self.nodes) +        # destinos finales (one-hot)
            num_agents * len(self.nodes)          # objetivos de tránsito (one-hot)
        )
        low = np.array([0] * obs_size, dtype=np.float32)
        # Construir el array high con los límites correctos para cada sección
        high_positions = [len(self.nodes) - 1] * num_agents
        high_water = [max_water] * num_agents
        high_fire = [max_fire] * len(self.fire_nodes)
        high_destinations = [1] * (num_agents * len(self.nodes))  # One-hot encoding (0 o 1)
        high_transit = [1] * (num_agents * len(self.nodes))       # One-hot encoding (0 o 1)
        high = np.array(
            high_positions + 
            high_water + 
            high_fire + 
            high_destinations + 
            high_transit,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.agent_positions = self.initial_agents_positions.copy()
        self.agent_water = [self.max_water for _ in range(self.num_agents)]
        
        # Resetear niveles de fuego según los atributos del grafo
        for node in self.fire_nodes:
            water_to_extinguish = self.graph.nodes[node].get('water_to_extinguish', self.max_fire)
            self.fire_levels[node] = water_to_extinguish

        self.final_destinations = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_transit_target = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_transit_source = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_in_transit = {agent_id: False for agent_id in range(self.num_agents)}
        self.agent_transit_time_remaining = {agent_id: 0 for agent_id in range(self.num_agents)}
            
        return self._get_obs()

    def _get_obs(self):
        # Posiciones actuales de los agentes
        pos_encoding = [self.nodes.index(pos) for pos in self.agent_positions]

        # One-hot encoding para los destinos finales de cada agente
        destination_encoding = []
        for agent_id in range(self.num_agents):
            dest = self.final_destinations[agent_id]
            # Vector de ceros con tamaño igual al número de nodos
            one_hot = [0] * len(self.nodes)
            if dest is not None:
                # Poner un 1 en la posición correspondiente al nodo destino
                one_hot[self.nodes.index(dest)] = 1
            destination_encoding.extend(one_hot)

        # One-hot encoding para los objetivos de tránsito de cada agente
        transit_target_encoding = []
        for agent_id in range(self.num_agents):
            target = self.agent_transit_target[agent_id]
            # Vector de ceros con tamaño igual al número de nodos
            one_hot = [0] * len(self.nodes)
            if target is not None:
                # Poner un 1 en la posición correspondiente al nodo objetivo
                one_hot[self.nodes.index(target)] = 1
            transit_target_encoding.extend(one_hot)

        # Resto de observaciones
        obs = np.array(
            pos_encoding + 
            self.agent_water + 
            list(self.fire_levels.values()) +
            destination_encoding +
            transit_target_encoding, 
            dtype=np.float32
        )

        return obs

    def step(self, actions):
        total_reward = 0

        for agent_id, target_node_idx in enumerate(actions):
            target_node = self.nodes[target_node_idx]
            #self.agent_positions[agent_id] = target_node
            self.move_to_node(agent_id, target_node)
            if self.agent_in_transit[agent_id]:
                # Si el agente está en tránsito, no actualizar la posición
                continue

            reward = 0

            if target_node in self.depot_nodes and self.agent_water[agent_id] < self.max_water:
                reward += (self.max_water - self.agent_water[agent_id]) * 0.5
                self.agent_water[agent_id] = self.max_water
            elif target_node in self.fire_nodes and self.fire_levels[target_node] > 0 and self.agent_water[agent_id] > 0:
                used_water = min(self.agent_water[agent_id], self.fire_levels[target_node])
                self.agent_water[agent_id] -= used_water
                self.fire_levels[target_node] -= used_water

                reward += (used_water) * 0.5
            else:
                reward -= 0.5  # penalización por inacción

            total_reward += reward

        done = all(level == 0 for level in self.fire_levels.values())

        return self._get_obs(), total_reward, done, {}
    
    def move_to_node(self, agent_id, target_node):
        """Move an agent to a target node with transit time."""
        # Si ya está en el nodo destino, no hacer nada
        if self.agent_positions[agent_id] == target_node:
            return 0
        # Si el agente ya está en tránsito, procesar el tiempo restante
        if self.agent_in_transit[agent_id]:
            return self.update_transit(agent_id, target_node)
        # Iniciar un nuevo tránsito
        current_node = self.agent_positions[agent_id]
        # Try to find path in the filtered graph
        try:
            path = nx.shortest_path(self.graph, source=current_node, target=target_node, weight='transit_time')
            # If path exists, proceed with movement
            self.final_destinations[agent_id] = target_node
            next_node = path[1] if len(path) > 1 else target_node
            transit_time = self.graph[current_node][next_node]['transit_time']
            # Registrar estado de tránsito
            self.agent_transit_target[agent_id] = next_node
            self.agent_transit_source[agent_id] = current_node
            self.agent_in_transit[agent_id] = True
            self.agent_transit_time_remaining[agent_id] = transit_time
            return 0
        except nx.NetworkXNoPath:
            # No path available with current width constraints
            return -0.5  # Small penalty for attempting an impossible move
        
    def update_transit(self, agent, target_node):
        # Reducir tiempo restante
        self.agent_transit_time_remaining[agent] -= 1
        # Si el tránsito ha terminado
        if self.agent_transit_time_remaining[agent] <= 0:
            # Si el agente llegó a su destino final
            if self.final_destinations[agent] == target_node:
                self.agent_positions[agent] = self.agent_transit_target[agent]
                self.agent_in_transit[agent] = False
                self.agent_transit_time_remaining[agent] = 0
                self.agent_transit_source[agent] = None
                self.agent_transit_target[agent] = None
                self.final_destinations[agent] = None
            # Si el agente llegó a un nodo intermedio
            else:
                # Actualizar posición del agente
                self.agent_positions[agent] = self.agent_transit_target[agent]
                # Si el objetivo no sigue siendo válido
                if self.final_destinations[agent] in self.fire_nodes:
                    # Si el incendio fue extinguido, actualizar el espacio de acción
                    if self.fire_levels[self.final_destinations[agent]] <= 0:
                        self.agent_in_transit[agent] = False
                        self.agent_transit_time_remaining[agent] = 0
                        self.agent_transit_source[agent] = None
                        self.agent_transit_target[agent] = None
                        self.final_destinations[agent] = None
                        return -0.5
                # Actualizar objetivo de tránsito
                self.agent_transit_source[agent] = self.agent_transit_target[agent]
                new_target_node = self.final_destinations[agent]
                try:
                    path = nx.shortest_path(self.graph, source=self.agent_transit_source[agent], target=new_target_node, weight='transit_time')
                    # Verificar si el camino tiene un solo nodo (origen y destino son iguales)
                    if len(path) <= 1:
                        # Ya estamos en el destino final
                        self.agent_positions[agent] = new_target_node
                        self.agent_in_transit[agent] = False
                        self.agent_transit_time_remaining[agent] = 0
                        self.agent_transit_source[agent] = None
                        self.agent_transit_target[agent] = None
                        self.final_destinations[agent] = None
                        return 0
                    # Si hay camino, continuar al siguiente nodo
                    self.agent_transit_time_remaining[agent] = self.graph[self.agent_transit_source[agent]][path[1]]['transit_time']
                    self.agent_transit_target[agent] = path[1]
                except nx.NetworkXNoPath:
                    print(f"Agent {agent} cannot reach target node {new_target_node} from {self.agent_transit_source[agent]}")
                    return -0.5  # Small penalty for attempting an impossible move
        return 0