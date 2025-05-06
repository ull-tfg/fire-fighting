import gym
from gym import spaces
import numpy as np
import networkx as nx
import random
# import pygame

class FirefightingEnv(gym.Env):
    def __init__(self, num_agents=3, graph=None, max_water=10, max_fire=50, vehicle_types=None):
        super(FirefightingEnv, self).__init__()
        self.num_agents = num_agents
        self.max_water = max_water
        self.max_fire = max_fire

        if vehicle_types is None:
            # Tipos predeterminados: {tipo: {width: ancho, max_water: capacidad}}
            self.vehicle_types = {
                'small': {'width': 1, 'max_water': 5},
                'medium': {'width': 2, 'max_water': 10},
                'large': {'width': 3, 'max_water': 15}
            }
        else:
            self.vehicle_types = vehicle_types
        # Asignar tipos de vehículos a los agentes (por defecto, distribuir equitativamente)
        self.agent_vehicle_types = {}
        vehicle_keys = list(self.vehicle_types.keys())
        for agent_id in range(num_agents):
            # Distribución cíclica de tipos de vehículos
            vehicle_type = vehicle_keys[agent_id % len(vehicle_keys)]
            self.agent_vehicle_types[agent_id] = vehicle_type

        # Inicializar capacidad de agua para cada agente según su tipo de vehículo
        self.agent_max_water = [self.vehicle_types[self.agent_vehicle_types[agent_id]]['max_water'] 
                              for agent_id in range(num_agents)]
        
        # Inicializar ocupación de aristas
        self.edge_occupancy = {}

        # Initialize transit-related dictionaries for all agents
        self.final_destinations = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_transit_target = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_transit_source = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_in_transit = {agent_id: False for agent_id in range(self.num_agents)}
        self.agent_transit_time_remaining = {agent_id: 0 for agent_id in range(self.num_agents)}
        
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
        if self.starter_nodes:
            self.initial_agents_positions = [self.starter_nodes[0] for _ in range(num_agents)]
        else:
            self.initial_agents_positions = [random.choice(self.nodes) for _ in range(num_agents)]
            
        self.agent_positions = self.initial_agents_positions.copy()
        self.agents_water = [self.agent_max_water[agent_id] for agent_id in range(num_agents)]

        # Inicializar ocupación de aristas
        for u, v in self.graph.edges():
            self.edge_occupancy[(u, v)] = []
            self.edge_occupancy[(v, u)] = []
        
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
            (num_agents * len(self.fire_nodes)) + (num_agents * len(self.depot_nodes)) +  # Matriz de distancias actuales completa 
            num_agents +                          # posiciones
            num_agents +                          # agua de agentes
            len(self.fire_nodes) +                # fuego en nodos
            num_agents * len(self.nodes) +        # destinos finales (one-hot)
            num_agents * len(self.nodes) +        # objetivos de tránsito (one-hot)
            num_agents +                          # estado de tránsito (1 transitando, 0 no)
            num_agents * len(self.vehicle_types)  # tipos de vehículos (one-hot)
        )
        low = np.array([0] * obs_size, dtype=np.float32)
        # Construir el array high con los límites correctos para cada sección
        high_distances = [50] * (num_agents * len(self.fire_nodes) + num_agents * len(self.depot_nodes))
        high_positions = [len(self.nodes) - 1] * num_agents
        high_water = [max_water] * num_agents
        high_fire = [max_fire] * len(self.fire_nodes)
        high_destinations = [1] * (num_agents * len(self.nodes))  # One-hot encoding (0 o 1)
        high_transit = [1] * (num_agents * len(self.nodes))       # One-hot encoding (0 o 1)
        high_transit_state = [1] * num_agents  # Estado de tránsito (0 o 1)
        high_vehicle_type = [1] * (num_agents * len(self.vehicle_types))  # One-hot encoding para tipos de vehículos
        high = np.array(
            high_distances +
            high_positions + 
            high_water + 
            high_fire + 
            high_destinations + 
            high_transit +
            high_transit_state +
            high_vehicle_type,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.agent_positions = self.initial_agents_positions.copy()
        # Usar la capacidad de agua específica del tipo de vehículo
        self.agents_water = [self.agent_max_water[agent_id] for agent_id in range(self.num_agents)]

        # Resetear niveles de fuego
        for node in self.fire_nodes:
            water_to_extinguish = self.graph.nodes[node].get('water_to_extinguish', self.max_fire)
            self.fire_levels[node] = water_to_extinguish

        # Resetear estados de tránsito
        self.final_destinations = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_transit_target = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_transit_source = {agent_id: None for agent_id in range(self.num_agents)}
        self.agent_in_transit = {agent_id: False for agent_id in range(self.num_agents)}
        self.agent_transit_time_remaining = {agent_id: 0 for agent_id in range(self.num_agents)}

        # Resetear ocupación de aristas
        for u, v in self.graph.edges():
            self.edge_occupancy[(u, v)] = []
            self.edge_occupancy[(v, u)] = []

        return self._get_obs()

    def _get_obs(self):
        # Distancias desde cada agente a los nodos clave
        distances_encoding = []

        for agent_id in range(self.num_agents):
            current_pos = self.agent_positions[agent_id]

            # Distancias a nodos de fuego usando subgrafos específicos para este agente
            for fire_node in self.fire_nodes:
                # Crear un subgrafo específico para este destino
                subgraph = self.create_subgraph(fire_node, agent_id)
                try:
                    dist = nx.shortest_path_length(subgraph, source=current_pos, target=fire_node, weight='transit_time')
                    distances_encoding.append(dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distances_encoding.append(50)  # Valor alto finito en lugar de infinito

            # Distancias a los depósitos
            for depot in self.depot_nodes:
                # Crear un subgrafo específico para este depósito
                subgraph = self.create_subgraph(depot, agent_id)
                try:
                    dist = nx.shortest_path_length(subgraph, source=current_pos, target=depot, weight='transit_time')
                    distances_encoding.append(dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distances_encoding.append(50)  # Valor alto finito

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

        # Estado de tránsito para cada agente (1=en tránsito, 0=no)
        transit_state = [1 if self.agent_in_transit[agent_id] else 0 for agent_id in range(self.num_agents)]

         # Añadir one-hot encoding para los tipos de vehículos
        vehicle_type_encoding = []
        vehicle_types_list = list(self.vehicle_types.keys())

        for agent_id in range(self.num_agents):
            agent_vehicle_type = self.agent_vehicle_types[agent_id]
            # Vector de ceros con tamaño igual al número de tipos de vehículos
            one_hot = [0] * len(self.vehicle_types)
            # Poner un 1 en la posición correspondiente al tipo de vehículo
            type_idx = vehicle_types_list.index(agent_vehicle_type)
            one_hot[type_idx] = 1
            vehicle_type_encoding.extend(one_hot)

        # Resto de observaciones
        obs = np.array(
            distances_encoding +
            pos_encoding + 
            self.agents_water + 
            list(self.fire_levels.values()) +
            destination_encoding +
            transit_target_encoding +
            transit_state +
            vehicle_type_encoding,
            dtype=np.float32
        )

        return obs

    def step(self, actions):
        """
        Execute one step for all agents.

        Args:
            actions: List of action indices, one per agent

        Returns:
            observations: Updated environment state
            rewards: Total reward for this step
            done: Whether the episode has ended
            info: Additional information
        """
        assert len(actions) == self.num_agents, f"Expected {self.num_agents} actions, got {len(actions)}"
        # Inicializar recompensas individuales y total
        agent_rewards = [0] * self.num_agents
        total_reward = 0
        info = {}
        # Tracking de agentes que completaron el movimiento en este turno
        just_arrived_agents = set()

        # Paso 1: Procesar movimiento de todos los agentes que están en tránsito
        for agent_id in range(self.num_agents):
            if self.agent_in_transit[agent_id]:
                reward = self.move_to_node(agent_id, self.agent_transit_target[agent_id])
                agent_rewards[agent_id] += reward
                # Si llegó a destino
                if not self.agent_in_transit[agent_id]:
                    if reward >= 0:
                        just_arrived_agents.add(agent_id)

        # Paso 2: Procesar acción automática para agentes que acaban de llegar
        for agent_id in just_arrived_agents:
            current_node = self.agent_positions[agent_id]
            reward = 0
            # Acciones automáticas según el tipo de nodo
            if current_node in self.depot_nodes:
                # Usar la capacidad máxima específica del vehículo
                max_water = self.agent_max_water[agent_id]
                if self.agents_water[agent_id] < max_water:
                    # Recompensa por recargar agua cuando se necesita
                    reward += (max_water - self.agents_water[agent_id]) * 0.5
                    self.agents_water[agent_id] = max_water
                else:
                    # Penalización por ir a un depósito con el tanque lleno
                    reward -= 1.0
            elif current_node in self.fire_nodes:
                if self.fire_levels[current_node] > 0:
                    if self.agents_water[agent_id] > 0:
                        # Recompensa por usar agua para apagar fuego
                        used_water = min(self.agents_water[agent_id], self.fire_levels[current_node])
                        self.agents_water[agent_id] -= used_water
                        self.fire_levels[current_node] -= used_water
                        reward += used_water * 0.5
                    else:
                        # Penalización por ir a un nodo de fuego sin agua
                        reward -= 1.0  # Penalización mayor por esta ineficiencia
            agent_rewards[agent_id] += reward

        # Paso 3: Procesar nuevas acciones para agentes que no están en tránsito
        for agent_id, action_idx in enumerate(actions):
            # Omitir agentes en tránsito (ya fueron procesados)
            if self.agent_in_transit[agent_id] or agent_id in just_arrived_agents:
                continue
            # Obtener el nodo objetivo para la acción
            target_node = self.nodes[action_idx]
            # Iniciar movimiento al nodo objetivo
            reward = 0
            reward = self.move_to_node(agent_id, target_node)
            # Si el agente ya está en el nodo destino, procesar acción automática
            if not self.agent_in_transit[agent_id] and agent_id not in just_arrived_agents:
                if target_node in self.depot_nodes:
                    # Usar la capacidad máxima específica del vehículo
                    max_water = self.agent_max_water[agent_id]
                    if self.agents_water[agent_id] < max_water:
                        # Recompensa por recargar agua cuando se necesita
                        reward += (max_water - self.agents_water[agent_id]) * 0.5
                        self.agents_water[agent_id] = max_water
                    else:
                        reward -= 1.0  
                elif target_node in self.fire_nodes:
                    if self.fire_levels[target_node] > 0:
                        if self.agents_water[agent_id] > 0:
                            used_water = min(self.agents_water[agent_id], self.fire_levels[target_node])
                            self.agents_water[agent_id] -= used_water
                            self.fire_levels[target_node] -= used_water
                            reward += used_water * 0.5
                        else:
                            # Penalización por ir a un nodo de fuego sin agua
                            reward -= 1.0
                    else:
                        # Penalización por ir a un fuego ya extinguido
                        reward -= 1.0
                else:
                    reward -= 0.5  # Penalización por inacción
            agent_rewards[agent_id] += reward

        # Calcular recompensa total y comprobar condición de terminación
        total_reward = sum(agent_rewards)
        done = all(level == 0 for level in self.fire_levels.values())
        # Penalizar cada paso para fomentar la eficiencia
        #for agent_id in range(self.num_agents):
        #    agent_rewards[agent_id] -= 0.1  # Penalización menor por cada paso

        # Añadir información adicional para depuración
        info = {
            'agent_rewards': agent_rewards,
            'agents_water': self.agents_water.copy(),
            'fire_levels': self.fire_levels.copy(),
            'fires_extinguished': sum(1 for level in self.fire_levels.values() if level == 0),
            'total_water_used': sum(self.max_water - water for water in self.agents_water)
        }

        return self._get_obs(), total_reward, done, info
    
    def get_closest_depot(self, agent_id):
        """Get the closest depot node to the agent."""
        current_pos = self.agent_positions[agent_id]
        min_distance = float('inf')
        closest_depot = None
        for depot in self.depot_nodes:
            # Crear un subgrafo específico para este depósito
            subgraph = self.create_subgraph(depot, agent_id)
            try:
                dist = nx.shortest_path_length(subgraph, source=current_pos, target=depot, weight='transit_time')
                if dist < min_distance:
                    min_distance = dist
                    closest_depot = depot
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return closest_depot

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
        
        immediate_reward = 0
        # Evaluar la calidad de la decisión para dar recompensa inmediata
        # Si el agente tiene poca agua y va a un depósito
        if target_node in self.depot_nodes and self.agents_water[agent_id] < self.agent_max_water[agent_id]/2:
            if target_node == self.get_closest_depot(agent_id):
                immediate_reward += 0.4 # Recompensa por ir al depósito más cercano
            else:
                immediate_reward -= 0.2 # Penalización por ir a un depósito más lejano
            immediate_reward += 0.4
        # Si el agente tiene agua y va a un nodo de incendio activo
        elif target_node in self.fire_nodes and self.fire_levels[target_node] > 0 and self.agents_water[agent_id] > 0:
            immediate_reward += 0.4
        # Penalización leve por decisiones subóptimas
        elif (target_node in self.depot_nodes and self.agents_water[agent_id] >= self.agent_max_water[agent_id]) or \
             (target_node in self.fire_nodes and self.fire_levels[target_node] <= 0) or \
             (target_node in self.fire_nodes and self.agents_water[agent_id] <= 0):
            immediate_reward -= 0.2

        # Crear un subgrafo para el agente
        subgraph = self.create_subgraph(target_node, agent_id)
        # Try to find path in the filtered graph
        try:
            path = nx.shortest_path(subgraph, source=current_node, target=target_node, weight='transit_time')
            # If path exists, proceed with movement
            self.final_destinations[agent_id] = target_node
            next_node = path[1] if len(path) > 1 else target_node
            transit_time = self.graph[current_node][next_node]['transit_time']
            # Registrar estado de tránsito
            self.agent_transit_target[agent_id] = next_node
            self.agent_transit_source[agent_id] = current_node
            self.agent_in_transit[agent_id] = True
            self.agent_transit_time_remaining[agent_id] = transit_time
            
            # Add agent to edge occupancy when starting to move (FIX: Add this line)
            self.add_edge_occupancy(current_node, next_node, agent_id)
            
            return immediate_reward
        except nx.NetworkXNoPath:
            # No path available with current width constraints
            return -0.5  # Small penalty for attempting an impossible move
        
    def update_transit(self, agent, target_node):
        # Reducir tiempo restante
        self.agent_transit_time_remaining[agent] -= 1

        # Si el tránsito ha terminado
        if self.agent_transit_time_remaining[agent] <= 0:
            # Eliminar agente de la arista
            self.remove_edge_occupancy(
                self.agent_transit_source[agent], 
                self.agent_transit_target[agent], 
                agent
            )
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
                # Create a subgraph for pathfinding
                subgraph = self.create_subgraph(new_target_node, agent)
                try:
                    path = nx.shortest_path(subgraph, source=self.agent_transit_source[agent], target=new_target_node, weight='transit_time')
                    # Si hay camino, continuar al siguiente nodo
                    self.agent_transit_time_remaining[agent] = self.graph[self.agent_transit_source[agent]][path[1]]['transit_time']
                    self.agent_transit_target[agent] = path[1]
                    # Actualizar ocupación de la arista
                    self.add_edge_occupancy(
                        self.agent_transit_source[agent], 
                        self.agent_transit_target[agent], 
                        agent
                    )
                except nx.NetworkXNoPath:
                    print(f"Agent {agent} cannot reach target node {new_target_node} from {self.agent_transit_source[agent]}")
                    return -0.5  # Small penalty for attempting an impossible move
        return 0
    
    def add_edge_occupancy(self, source, target, agent):
        """Set the occupancy of an edge by an agent."""
        edge = (source, target)
        reverse_edge = (target, source) # Ocupación de la arista en ambas direcciones
        self.edge_occupancy[edge].append(agent)
        self.edge_occupancy[reverse_edge].append(agent)
    
    def remove_edge_occupancy(self, source, target, agent):
        """Remove the occupancy of an edge by an agent."""
        edge = (source, target)
        reverse_edge = (target, source)
        self.edge_occupancy[edge].remove(agent)
        self.edge_occupancy[reverse_edge].remove(agent)
    
    def edge_space_left(self, source, target):
        """Check how much space is left on an edge."""
        edge = (source, target)
        # FIX: Correctly access the vehicle width through self.vehicle_types
        occupancy = sum(self.vehicle_types[self.agent_vehicle_types[agent]]['width'] 
                       for agent in self.edge_occupancy[edge])

        space_left = self.graph[source][target].get('width', 1) - occupancy
        return space_left

    def can_transit(self, source, target, agent):
        """Check if the agent can transit through an edge."""
        # FIX: Correctly access the vehicle width through self.vehicle_types
        agent_vehicle_type = self.agent_vehicle_types[agent]
        agent_width = self.vehicle_types[agent_vehicle_type]['width']

        if agent_width > self.edge_space_left(source, target):
            return False
        return True
    
    def create_subgraph(self, target_node, agent):
        """Create a subgraph for pathfinding."""
        # Create a subgraph with the base edge occupancy where the agent can move
        subgraph = self.graph.copy()

        # Guardar la posición actual del agente para evitar quedar atrapado
        agent_current_position = self.agent_positions[agent]
        # Eliminar aristas que conectan incendios activos entre sí
        active_fire_nodes = [node for node in self.fire_nodes 
                             if node in self.fire_levels and self.fire_levels[node] > 0]
        for fire_node1 in active_fire_nodes:
            for fire_node2 in active_fire_nodes:
                if fire_node1 != fire_node2 and subgraph.has_edge(fire_node1, fire_node2):
                    subgraph.remove_edge(fire_node1, fire_node2)
        # Eliminar conexiones de nodos de incendio activo (excepto casos especiales)
        for fire_node in active_fire_nodes:
            # Skip the target node (siempre permitir ir hacia el destino)
            if fire_node == target_node:
                # Solo mantener conexiones al destino desde nodos no-incendio
                for neighbor in list(subgraph.neighbors(fire_node)):
                    if neighbor in active_fire_nodes and neighbor != agent_current_position:
                        subgraph.remove_edge(fire_node, neighbor)
                continue
            # No eliminar conexiones si el agente está actualmente en este nodo
            if fire_node == agent_current_position:
                # Permitir salir pero solo hacia nodos que no son incendios
                for neighbor in list(subgraph.neighbors(fire_node)):
                    if neighbor in active_fire_nodes and neighbor != target_node:
                        subgraph.remove_edge(fire_node, neighbor)
                continue
            # Eliminar todas las conexiones de este incendio activo
            for neighbor in list(subgraph.neighbors(fire_node)):
                if subgraph.has_edge(fire_node, neighbor):
                    subgraph.remove_edge(fire_node, neighbor)

        # Remove edges with lower width than the agent's vehicle width
        # FIX: Correctly access the vehicle width through self.vehicle_types
        agent_vehicle_type = self.agent_vehicle_types[agent]
        agent_width = self.vehicle_types[agent_vehicle_type]['width']

        for edge_tuple, agents in self.edge_occupancy.items():
            if edge_tuple[0] in subgraph and edge_tuple[1] in subgraph:
                if subgraph.has_edge(edge_tuple[0], edge_tuple[1]):
                    if agent_width > self.graph[edge_tuple[0]][edge_tuple[1]].get('width', 1):
                        subgraph.remove_edge(edge_tuple[0], edge_tuple[1])

        return subgraph
    
    def node_is_reachable(self, source_node, target_node, agent):
        """Check if a node is reachable from the agent's current position."""
        # Check if the target node is reachable from the source node
        return nx.has_path(self.create_subgraph(target_node, agent), source=source_node, target=target_node)
