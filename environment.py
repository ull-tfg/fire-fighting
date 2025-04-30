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
        if self.starter_nodes:
            self.initial_agents_positions = [self.starter_nodes[0] for _ in range(num_agents)]
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
            (num_agents * len(self.fire_nodes)) + num_agents +  # distances_encoding (distancias a fuegos + depósito)
            num_agents +                          # posiciones
            num_agents +                          # agua de agentes
            len(self.fire_nodes) +                # fuego en nodos
            num_agents * len(self.nodes) +        # destinos finales (one-hot)
            num_agents * len(self.nodes) +        # objetivos de tránsito (one-hot)
            num_agents                            # estado de tránsito (1 transitando, 0 no)
        )
        low = np.array([0] * obs_size, dtype=np.float32)
        # Construir el array high con los límites correctos para cada sección
        high_distances = [20] * (num_agents * len(self.fire_nodes) + num_agents)  # Asumiendo distancias máximas de 20
        high_positions = [len(self.nodes) - 1] * num_agents
        high_water = [max_water] * num_agents
        high_fire = [max_fire] * len(self.fire_nodes)
        high_destinations = [1] * (num_agents * len(self.nodes))  # One-hot encoding (0 o 1)
        high_transit = [1] * (num_agents * len(self.nodes))       # One-hot encoding (0 o 1)
        high_transit_state = [1] * num_agents  # Estado de tránsito (0 o 1)
        high = np.array(
            high_distances +
            high_positions + 
            high_water + 
            high_fire + 
            high_destinations + 
            high_transit +
            high_transit_state,
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
        # Distancias desde cada agente a los nodos clave
        distances_encoding = []
        for agent_id in range(self.num_agents):
            current_pos = self.agent_positions[agent_id]
            # Distancias a nodos de fuego
            for fire_node in self.fire_nodes:
                try:
                    dist = nx.shortest_path_length(self.graph, current_pos, fire_node, weight='transit_time')
                    distances_encoding.append(dist)
                except:
                    distances_encoding.append(float('inf'))
            # Distancia al depósito más cercano
            depot_dists = []
            for depot in self.depot_nodes:
                try:
                    depot_dists.append(nx.shortest_path_length(self.graph, current_pos, depot, weight='transit_time'))
                except:
                    depot_dists.append(float('inf'))
            if depot_dists:
                min_depot_dist = min(depot_dists)
                distances_encoding.append(min_depot_dist)
            else:
                distances_encoding.append(float('inf'))

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

        # Resto de observaciones
        obs = np.array(
            distances_encoding +
            pos_encoding + 
            self.agent_water + 
            list(self.fire_levels.values()) +
            destination_encoding +
            transit_target_encoding +
            transit_state,
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
                if self.agent_water[agent_id] < self.max_water:
                    # Recompensa por recargar agua cuando se necesita
                    reward += (self.max_water - self.agent_water[agent_id]) * 0.5
                    self.agent_water[agent_id] = self.max_water
                else:
                    # Penalización por ir a un depósito con el tanque lleno
                    reward -= 1.0  # Penalización significativa por ineficiencia
            elif current_node in self.fire_nodes:
                if self.fire_levels[current_node] > 0:
                    if self.agent_water[agent_id] > 0:
                        # Recompensa por usar agua para apagar fuego
                        used_water = min(self.agent_water[agent_id], self.fire_levels[current_node])
                        self.agent_water[agent_id] -= used_water
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
            self.move_to_node(agent_id, target_node)
            # Si el agente ya está en el nodo destino, procesar acción automática
            if not self.agent_in_transit[agent_id] and agent_id not in just_arrived_agents:
                reward = 0
                if target_node in self.depot_nodes:
                    if self.agent_water[agent_id] < self.max_water:
                        reward += (self.max_water - self.agent_water[agent_id]) * 0.5
                        self.agent_water[agent_id] = self.max_water
                    else:
                        # Penalización por ir a un depósito con el tanque lleno
                        reward -= 1.0
                elif target_node in self.fire_nodes:
                    if self.fire_levels[target_node] > 0:
                        if self.agent_water[agent_id] > 0:
                            used_water = min(self.agent_water[agent_id], self.fire_levels[target_node])
                            self.agent_water[agent_id] -= used_water
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
            'agent_water': self.agent_water.copy(),
            'fire_levels': self.fire_levels.copy(),
            'fires_extinguished': sum(1 for level in self.fire_levels.values() if level == 0),
            'total_water_used': sum(self.max_water - water for water in self.agent_water)
        }

        return self._get_obs(), total_reward, done, info
    
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
                    # Si hay camino, continuar al siguiente nodo
                    self.agent_transit_time_remaining[agent] = self.graph[self.agent_transit_source[agent]][path[1]]['transit_time']
                    self.agent_transit_target[agent] = path[1]
                except nx.NetworkXNoPath:
                    print(f"Agent {agent} cannot reach target node {new_target_node} from {self.agent_transit_source[agent]}")
                    return -0.5  # Small penalty for attempting an impossible move
        return 0