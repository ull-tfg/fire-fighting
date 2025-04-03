import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from enum import Enum
import random
import scipy as sp
from collections import defaultdict

class FirefightingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, graph, num_agents=2, max_steps=200, vehicle_types=None):
        """
        Initialize the Firefighting Environment.
        
        Args:
            graph: NetworkX graph representing the environment. If None, a random graph will be generated.
            num_fires: Number of fires to generate if graph is None.
            num_water_sources: Number of water sources to generate if graph is None.
            num_agents: Number of firefighting agents.
            max_steps: Maximum number of steps per episode.
            vehicle_types: Dict mapping agent_id to vehicle type properties {'capacity': int, 'width': int}.
        """
        super().__init__()
        
        # Environment parameters
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        # Set up the graph
        self.graph = graph
        # Extract nodes by type
        self.fire_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['type'] == 'fire']
        self.tank_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['type'] == 'tank']
        self.starter_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['type'] == 'starter']
        self.all_nodes = list(self.graph.nodes())
        # Node mapping for observation space
        self.node_to_idx = {node: i for i, node in enumerate(self.all_nodes)}
        # Vehicle setup
        self.default_vehicle = {'capacity': 200, 'width': 2}
        if vehicle_types is None:
            self.vehicle_types = {i: self.default_vehicle for i in range(num_agents)}
        else:
            self.vehicle_types = vehicle_types
            
        # Agent state
        self.agent_positions = {}              # Current node positions
        self.agent_water_levels = {}           # Current water levels
        self.agent_in_transit = {}             # Whether agent is currently moving between nodes
        self.agent_transit_time_remaining = {} # Time remaining until reaching destination
        self.agent_transit_source = {}         # Source node of current transit
        self.agent_transit_target = {}         # Target node of current transit
        self.edge_occupancy = {}               # Track which agents are on which edges
        self.final_destinations = {}           # Final destination for each agent

        self.fires_remaining = {}  # Maps fire nodes to remaining water needed
        
        
        # Define observation space (one per agent)
        # State includes:
        # - Full adjacency matrix of the graph (len(all_nodes) * len(all_nodes))
        # - Active fire nodes (1 if fire is present, 0 otherwise) (len(all_nodes))
        # - Water source nodes (1 if it's a reservoir, 0 if not) (len(all_nodes))
        # - One-hot encoding of the agent's position (len(all_nodes))
        # - Agent's current water level (1)
        # - One-hot encoding of all other agents' positions (len(all_nodes) * (num_agents-1))
        # - Transit status (1)
        # - Transit time remaining (1)
        # - Agent's next destination (len(all_nodes))
        # - Agent's final destination (len(all_nodes))
        # - Other agents' next destinations (len(all_nodes) * (num_agents-1))
        # - Other agents' final destinations (len(all_nodes) * (num_agents-1))
        self.state_dim = (
            len(self.all_nodes) * len(self.all_nodes) +  # Full adjacency matrix
            len(self.all_nodes) +    # Active fire nodes
            len(self.all_nodes) +    # Water source nodes
            len(self.all_nodes) +    # One-hot encoding of agent's position
            self.num_agents + # Current water level of all agents
            len(self.all_nodes) * (self.num_agents - 1) +  # Other agents' positions
            self.num_agents +        # Transit time remaining of all agents
            len(self.all_nodes) +    # Agent's next destination
            len(self.all_nodes) +    # Agent's final destination
            len(self.all_nodes) * (self.num_agents - 1) +  # Other agents' next destinations
            len(self.all_nodes) * (self.num_agents - 1)    # Other agents' final destinations
        )

        # Define action space (one per agent)
        self.fire_actions = self.fire_nodes.copy()  # Actions to target fires
        self.tank_actions = self.tank_nodes.copy()  # Actions to target tanks
        self.all_actions = self.fire_actions + self.tank_actions  # Combined actions
        # Espacios de acción individuales para cada agente
        self.agent_action_spaces = {}
        for agent_id in range(self.num_agents):
            self.agent_action_spaces[agent_id] = {
                'fires': self.fire_actions.copy(),
                'tanks': self.tank_actions.copy(),
                'available': self.all_actions.copy()
            }
        # Para compatibilidad con gymnasium
        self.action_space = spaces.Tuple([
            spaces.Discrete(len(self.all_actions))  # Tamaño máximo posible del espacio de acciones
            for _ in range(self.num_agents)
        ])
        
        low = np.zeros(self.state_dim, dtype=np.float32)
        high = np.ones(self.state_dim, dtype=np.float32) * float('inf')
        self.observation_space = spaces.Tuple([
            spaces.Box(low=low, high=high, dtype=np.float32)
            for _ in range(self.num_agents)
        ])
        
        # Statistics tracking
        self.total_reward = 0
        self.fires_extinguished = 0
        self.agent_rewards = {i: 0 for i in range(self.num_agents)}
        
        # Initialize the state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_reward = 0
        self.fires_extinguished = 0
        self.agent_rewards = {i: 0 for i in range(self.num_agents)}
        
        # Reset fire statuses
        self.fires_remaining = {
            node: self.graph.nodes[node]['water_to_extinguish'] 
            for node in self.fire_nodes
        }
        
        # Reset agent positions and water levels
        self.agent_positions = {}
        self.agent_water_levels = {}
        self.agent_in_transit = {}
        self.agent_transit_time_remaining = {}
        self.agent_transit_source = {}
        self.agent_transit_target = {}
        self.edge_occupancy = {}  # Initialize empty edge occupancy
        self.final_destinations = {}

        # Initialize edge occupancy for all edges
        for edge in self.graph.edges():
            self.edge_occupancy[edge] = []
            self.edge_occupancy[(edge[1], edge[0])] = []  # Also add reverse direction
        
        for agent_id in range(self.num_agents):
            available_nodes = [node for node in self.starter_nodes]
            # start_node = random.choice(available_nodes)  # Randomly select a starting node
            start_node = available_nodes[0] # All agents start at the same node for consistency in results (steps done)
            self.agent_positions[agent_id] = start_node
            
            # Start with all water
            capacity = self.vehicle_types[agent_id]['capacity']
            self.agent_water_levels[agent_id] = capacity

            # Initialize transit state
            self.agent_in_transit[agent_id] = False
            self.agent_transit_time_remaining[agent_id] = 0
            self.agent_transit_source[agent_id] = None
            self.agent_transit_target[agent_id] = None
        
        # Get first observation for all agents
        observations = tuple(self._get_observation(agent_id) for agent_id in range(self.num_agents))
        info = self._get_info()

        # Update action spaces for all agents
        for agent_id in range(self.num_agents):
            self._update_agent_action_space(agent_id)
        
        return observations, info
    
    def step(self, actions):
        """
        Execute one step for all agents.
        
        Args:
            actions: Tuple of actions, one per agent
        
        Returns:
            observations: Tuple of observations, one per agent
            rewards: Tuple of rewards, one per agent
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated
            info: Additional information
        """
        assert len(actions) == self.num_agents, f"Expected {self.num_agents} actions, got {len(actions)}"
        rewards = [0] * self.num_agents
        # Tracking de agentes que completaron el movimiento en este turno
        just_arrived_agents = set()

        # 1. Procesar movimiento de todos los agentes
        for agent_id in range(self.num_agents):
            if self.agent_in_transit[agent_id]:
                reward = self.move_to_node(agent_id, self.agent_transit_target[agent_id])
                rewards[agent_id] += reward
                self.agent_rewards[agent_id] += reward
                # Si llegó a destino en este turno
                if not self.agent_in_transit[agent_id]:
                    if reward >= 0:
                        just_arrived_agents.add(agent_id)
                    # Actualizar espacio de acciones pero NO procesar acciones automáticas todavía
                    self._update_agent_action_space(agent_id)

        # 2. Procesar llegadas
        for agent_id in just_arrived_agents:
            arrived_node = self.agent_positions[agent_id]
            # Si el nodo de llegada no es un nodo de fuego o tanque, actualizar espacio de acción
            if arrived_node not in self.fire_nodes and arrived_node not in self.tank_nodes:
                self._update_agent_action_space(agent_id)
                continue
            # Si el agente llegó a un nodo de fuego o tanque, procesar acción
            action = self.all_actions.index(arrived_node)
            reward = self._process_agent_action(agent_id, action)
            rewards[agent_id] += reward
            self.agent_rewards[agent_id] += reward

        # 3. Elección de acción para agentes que NO llegaron en este turno
        for agent_id, action in enumerate(actions):
            # Omitir agentes que están en tránsito o acaban de llegar
            if self.agent_in_transit[agent_id] or agent_id in just_arrived_agents:
                continue
            # Procesar acción normalmente
            target_node = self.all_actions[action]
            if target_node not in self.agent_action_spaces[agent_id]['available']:
                rewards[agent_id] = 0
                continue
            reward = self._process_agent_action(agent_id, action)
            rewards[agent_id] += reward
            self.agent_rewards[agent_id] += reward

        # UPDATE global reward
        # Advance step and check episode completion
        self.current_step += 1
        # Terminate if all fires are extinguished
        terminated = all(fire <= 0 for fire in self.fires_remaining.values())
        truncated = self.current_step >= self.max_steps
        # Penalize each agent for each step taken
        for agent_id in range(self.num_agents):
            # Penalize for each step taken
            rewards[agent_id] -= 0.3
            self.agent_rewards[agent_id] -= 0.3
        # Bonus for extinguishing all fires
        if terminated:
            for agent_id in range(self.num_agents):
                rewards[agent_id] += 25
                self.agent_rewards[agent_id] += 25

        self.total_reward += sum(rewards)
        # Get observations for all agents
        observations = tuple(self._get_observation(agent_id) for agent_id in range(self.num_agents))
        info = self._get_info()
        
        return observations, tuple(rewards), terminated, truncated, info
    
    def _update_agent_action_space(self, agent_id):
        """Actualiza el espacio de acciones para un agente basado en su posición actual."""
        available_actions = []
        # Si está en tránsito, no puede hacer nada
        if self.agent_in_transit[agent_id]:
            self.agent_action_spaces[agent_id]['available'] = []
            return

        # Incendios activos si tiene agua
        if self.agent_water_levels[agent_id] > 0:
            for fire_node in self.fire_actions:
                # Verificar que el incendio aún existe o si el agente puede llegar a él
                if fire_node not in self.fires_remaining or self.fires_remaining[fire_node] <= 0 or self.node_is_reachable(self.agent_positions[agent_id], fire_node, agent_id) is False:
                    continue
                # Verificar si ya hay otro agente dirigiéndose a este incendio
                agents_targeting_this_fire = [
                    a for a in range(self.num_agents) 
                    if self.agent_in_transit[a] and self.final_destinations[a] == fire_node and a != agent_id
                ]
                if agents_targeting_this_fire:
                    # Calcular agua necesaria y agua en camino
                    water_needed = self.fires_remaining[fire_node]
                    water_on_the_way = sum(
                        min(self.agent_water_levels[a], water_needed) 
                        for a in agents_targeting_this_fire
                    )
                    # Solo añadir el fuego si se necesita más agua
                    if water_needed > water_on_the_way:
                        available_actions.append(fire_node)
                    else:
                        # Check if current agent can arrive faster than the slowest agent already headed there
                        # Create a temporary subgraph for path finding
                        temp_graph = self.graph.copy()
                        for u, v, data in list(temp_graph.edges(data=True)):
                            if not self.can_transit(u, v, agent_id):
                                temp_graph.remove_edge(u, v)
                        try:
                            # Calculate time for current agent to reach fire
                            path = nx.shortest_path(temp_graph, source=self.agent_positions[agent_id], 
                                                    target=fire_node, weight='transit_time')
                            agent_time_to_arrive = sum(self.graph[path[i]][path[i+1]]['transit_time'] 
                                                      for i in range(len(path)-1))                           
                            # Find slowest agent already targeting this fire
                            slowest_time_to_arrive = float('-inf')
                            for a in agents_targeting_this_fire:
                                remaining_time = self.agent_transit_time_remaining[a]
                                if self.final_destinations[a] != self.agent_transit_target[a]:
                                    agent_graph = self.graph.copy()
                                    for u, v, data in list(agent_graph.edges(data=True)):
                                        if not self.can_transit(u, v, a):
                                            agent_graph.remove_edge(u, v)
                                    remaining_path = nx.shortest_path(agent_graph, source=self.agent_transit_target[a],
                                                                     target=self.final_destinations[a], weight='transit_time')
                                    remaining_time += sum(self.graph[remaining_path[i]][remaining_path[i+1]]['transit_time']
                                                         for i in range(len(remaining_path)-1))
                                slowest_time_to_arrive = max(slowest_time_to_arrive, remaining_time)
                            # If current agent can arrive faster, add the fire node
                            if agent_time_to_arrive < slowest_time_to_arrive:
                                available_actions.append(fire_node)
                        except nx.NetworkXNoPath:
                            # Current agent can't reach the fire
                            pass
                else:
                    # Si no hay otros agentes, añadir el fuego
                    available_actions.append(fire_node)
                
        # Añadir tanques si necesita agua
        if self.agent_water_levels[agent_id] < self.vehicle_types[agent_id]['capacity']:
            for tank_node in self.tank_actions:
                if self.node_is_reachable(self.agent_positions[agent_id], tank_node, agent_id) is True:
                    available_actions.append(tank_node)

        # Actualizar el espacio de acciones del agente
        self.agent_action_spaces[agent_id]['available'] = available_actions
    
    def _process_agent_action(self, agent_id, action):
        """Process an individual agent's action and return the reward."""
        # Get the actual target node based on the action index
        target_node = self.all_actions[action]
        # Determinar si es un nodo de fuego o de tanque
        if target_node in self.fire_nodes:
            # Lógica para manejar acción en un nodo de fuego
            return self.handle_fire_action(agent_id, target_node)
        elif target_node in self.tank_nodes:
            # Lógica para manejar acción en un nodo de tanque
            return self.handle_tank_action(agent_id, target_node)
        # Manejar caso inesperado
        raise ValueError(f"Acción no válida: {action}")
    
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
        """Check if an edge is occupied by any agent."""
        edge = (source, target)
        occupancy = sum(self.vehicle_types[agent]['width'] for agent in self.edge_occupancy[edge])
        space_left = self.graph[source][target]['width'] - occupancy
        return space_left
    
    def can_transit(self, source, target, agent):
        """Update edge occupancy."""
        agent_width = self.vehicle_types[agent]['width']
        if agent_width > self.edge_space_left(source, target):
            return False
        return True
    
    def node_is_reachable(self, source_node, target_node, agent):
        """Check if a node is reachable from the agent's current position."""
        # Create a subgraph with the base edge occupancy where the agent can move
        subgraph = self.graph.copy()
        # Remove edges with lower width than the agent's vehicle width
        for edge_tuple, agents in self.edge_occupancy.items():
            # Check if the edge exists in the graph before accessing
            if edge_tuple in self.graph.edges():
                if self.vehicle_types[agent]['width'] > self.graph[edge_tuple[0]][edge_tuple[1]]['width']:
                    # Edge is too narrow for this vehicle
                    if subgraph.has_edge(edge_tuple[0], edge_tuple[1]):
                        subgraph.remove_edge(edge_tuple[0], edge_tuple[1])

        # Check if the target node is reachable from the source node
        try:
            return nx.has_path(subgraph, source=source_node, target=target_node)
        except nx.NetworkXNoPath:
            return False

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
                    if self.fires_remaining[self.final_destinations[agent]] <= 0:
                        self.agent_in_transit[agent] = False
                        self.agent_transit_time_remaining[agent] = 0
                        self.agent_transit_source[agent] = None
                        self.agent_transit_target[agent] = None
                        self.final_destinations[agent] = None
                        return -20
                # Actualizar objetivo de tránsito
                self.agent_transit_source[agent] = self.agent_transit_target[agent]
                new_target_node = self.final_destinations[agent]
                # Create a temporary subgraph excluding edges that are too narrow
                temp_graph = self.graph.copy()
                for u, v, data in list(temp_graph.edges(data=True)):
                    # Skip if edge is too narrow for this agent's vehicle
                    if not self.can_transit(u, v, agent):
                        temp_graph.remove_edge(u, v)
                try:
                    path = nx.shortest_path(temp_graph, source=self.agent_transit_source[agent], target=new_target_node, weight='transit_time')
                    self.agent_transit_time_remaining[agent] = self.graph[self.agent_transit_source[agent]][path[1]]['transit_time']
                    self.agent_transit_target[agent] = path[1]
                    # Actualizar ocupación de aristas
                    self.add_edge_occupancy(
                        self.agent_transit_source[agent],
                        self.agent_transit_target[agent],
                        agent
                    )
                except nx.NetworkXNoPath:
                    print(f"Agent {agent} cannot reach target node {new_target_node} from {self.agent_transit_source[agent]}")
                    return -5  # Small penalty for attempting an impossible move
        return 0
        
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
        # Create a temporary subgraph excluding edges that are too narrow
        temp_graph = self.graph.copy()
        for u, v, data in list(temp_graph.edges(data=True)):
            # Skip if edge is too narrow for this agent's vehicle
            if not self.can_transit(u, v, agent_id):
                temp_graph.remove_edge(u, v)
        # Try to find path in the filtered graph
        try:
            path = nx.shortest_path(temp_graph, source=current_node, target=target_node, weight='transit_time')
            # If path exists, proceed with movement
            self.final_destinations[agent_id] = target_node
            next_node = path[1] if len(path) > 1 else target_node
            transit_time = self.graph[current_node][next_node]['transit_time']
            # Registrar estado de tránsito
            self.agent_transit_target[agent_id] = next_node
            self.agent_transit_source[agent_id] = current_node
            self.agent_in_transit[agent_id] = True
            self.agent_transit_time_remaining[agent_id] = transit_time
            # Actualizar ocupación de aristas
            self.add_edge_occupancy(current_node, next_node, agent_id)
            return 0
        except nx.NetworkXNoPath:
            # No path available with current width constraints
            return -5  # Small penalty for attempting an impossible move
    
    def handle_fire_action(self, agent_id, fire_node):
        """Handle an agent's action to target a fire node."""
        # Check if fire still exists
        if fire_node not in self.fires_remaining or self.fires_remaining[fire_node] <= 0:
            return -20  # Penalización por intentar extinguir un incendio que ya no existe
        # Check if agent has enough water
        if self.agent_water_levels[agent_id] <= 0:
            return 0
        # Si el agente no está en el nodo de incendio, moverse hacia él
        if self.agent_positions[agent_id] != fire_node:
            self.move_to_node(agent_id, fire_node)
            if self.agent_in_transit[agent_id]:
                return 0  # No reward if agent is in transit
            
        # Try to extinguish the fire
        water_needed = self.fires_remaining[fire_node]
        water_available = self.agent_water_levels[agent_id]
        water_used = min(water_needed, water_available)
        self.fires_remaining[fire_node] -= water_used
        self.agent_water_levels[agent_id] -= water_used
        # Update rewards
        reward = water_used * 0.1  # Reward for water used
        if self.fires_remaining[fire_node] <= 0:
            reward += 25  # Bonus for extinguishing fire
            self.fires_extinguished += 1
            # Si el incendio fue extinguido, actualizar los espacios de acción de TODOS los agentes
            for aid in range(self.num_agents):
                self._update_agent_action_space(aid)
        else:
            # Solo actualizar el espacio de acción del agente actual
            self._update_agent_action_space(agent_id)

        return reward
        
    def handle_tank_action(self, agent_id, tank_node):
        """Handle an agent's action to target a water tank node."""
        # Si el agente no está en el nodo del tanque, moverse hacia él
        if self.agent_positions[agent_id] != tank_node:
            self.move_to_node(agent_id, tank_node)
            # Si comenzó a moverse, no puede recargar agua aún
            if self.agent_in_transit[agent_id]:
                return 0
        # Refill water
        capacity = self.vehicle_types[agent_id]['capacity']
        self.agent_water_levels[agent_id] = capacity
        # Update action space
        self._update_agent_action_space(agent_id)
        return 0
        
    
    def _get_observation(self, agent_id):
        """
        Generate the observation for a specific agent.

        Args:
            agent_id: ID of the agent to generate observation for

        Returns:
            numpy.ndarray: Observation vector
        """
        # Full adjacency matrix for the graph (1 if connected, 0 if not connected)
        adjacency_matrix = np.zeros((len(self.all_nodes) * len(self.all_nodes)), dtype=np.float32)

        # For each node, set 1 for all its neighbors
        for i, node in enumerate(self.all_nodes):
            for neighbor in self.graph.neighbors(node):
                neighbor_idx = self.node_to_idx[neighbor]
                adjacency_matrix[i * len(self.all_nodes) + neighbor_idx] = 1.0

        # Active fire nodes (water needed to extinguish)
        fire_status = np.array([
            self.fires_remaining[node]
            if node in self.fires_remaining else 0.0
            for node in self.all_nodes
        ], dtype=np.float32)

        # Water source nodes (1 if it's a reservoir, 0 if not)
        water_nodes = np.array([1 if node in self.tank_nodes else 0 for node in self.all_nodes], dtype=np.float32)

        # Agent's position (one-hot encoding)
        agent_pos_encoding = np.zeros(len(self.all_nodes), dtype=np.float32)
        pos_idx = self.node_to_idx[self.agent_positions[agent_id]]
        agent_pos_encoding[pos_idx] = 1.0

        # Agents water level
        water_level = np.array([self.agent_water_levels[agent_id]], dtype=np.float32)
        # Other agents' water levels
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                # Current water level
                other_water_level = self.agent_water_levels[other_id]
                # Add to the observation
                water_level = np.append(water_level, other_water_level)

        # Other agents' positions (one-hot encoding for each)
        other_agents_pos = np.zeros(len(self.all_nodes) * (self.num_agents - 1), dtype=np.float32)
        # Other agents' next destinations (in transit)
        other_agents_next_dest = np.zeros(len(self.all_nodes) * (self.num_agents - 1), dtype=np.float32)
        # Other agents' final destinations (planned target)
        other_agents_final_dest = np.zeros(len(self.all_nodes) * (self.num_agents - 1), dtype=np.float32)

        # Process other agents' positions and destinations
        other_idx = 0
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                # Current position
                other_pos = self.agent_positions[other_id]
                other_pos_idx = self.node_to_idx[other_pos]
                other_agents_pos[other_idx * len(self.all_nodes) + other_pos_idx] = 1.0

                # If in transit, add next destination (immediate target)
                if self.agent_in_transit[other_id]:
                    next_dest = self.agent_transit_target[other_id]
                    next_dest_idx = self.node_to_idx[next_dest]
                    other_agents_next_dest[other_idx * len(self.all_nodes) + next_dest_idx] = 1.0

                    final_dest = self.final_destinations[other_id]
                    final_dest_idx = self.node_to_idx[final_dest]
                    other_agents_final_dest[other_idx * len(self.all_nodes) + final_dest_idx] = 1.0

                other_idx += 1

        # Agent's own transit information
        # Next destination (one-hot encoding)
        agent_next_dest = np.zeros(len(self.all_nodes), dtype=np.float32)

        # Final destination (one-hot encoding)
        agent_final_dest = np.zeros(len(self.all_nodes), dtype=np.float32)

        # All agents transit time remaining
        transit_time = np.array([self.agent_transit_time_remaining[agent_id]], dtype=np.float32)
        # Other agents' transit time remaining
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                # Current water level
                other_transit_time = self.agent_transit_time_remaining[other_id]
                # Add to the observation
                transit_time = np.append(transit_time, other_transit_time)

        # If agent is in transit, set destinations
        if self.agent_in_transit[agent_id]:
            # Next/immediate destination
            transit_target = self.agent_transit_target[agent_id]
            target_idx = self.node_to_idx[transit_target]
            final_idx = self.node_to_idx[self.final_destinations[agent_id]]
            agent_next_dest[target_idx] = 1.0
            agent_final_dest[final_idx] = 1.0

        # Concatenate all features into a single observation vector
        observation = np.concatenate([
            adjacency_matrix,      # Full graph adjacency matrix
            fire_status,           # Active fires
            water_nodes,           # Water sources
            agent_pos_encoding,    # Agent's position
            water_level,           # Agent's water level
            other_agents_pos,      # Other agents' positions
            transit_time,          # Agent's transit time remaining
            agent_next_dest,       # Agent's next destination
            agent_final_dest,      # Agent's final destination
            other_agents_next_dest,# Other agents' next destinations
            other_agents_final_dest# Other agents' final destinations
        ])

        return observation
    
    def _get_info(self):
        """Return additional information about the current state."""
        return {
            'fires_remaining': sum(1 for v in self.fires_remaining.values() if v > 0),
            'fires_extinguished': self.fires_extinguished,
            'total_reward': self.total_reward,
            'agent_rewards': self.agent_rewards,
            'steps': self.current_step,
            'agent_positions': self.agent_positions,
            'agent_water_levels': self.agent_water_levels,
            'agent_in_transit': self.agent_in_transit,
            'agent_transit_time': self.agent_transit_time_remaining,
            'edge_occupancy': {edge: len(agents) for edge, agents in self.edge_occupancy.items() if agents}
        }
    
    def render(self):
        """Render the environment."""
        # Implementation would depend on desired visualization
        # For now, we'll just print basic information for all agents
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Fires left: {sum(1 for v in self.fires_remaining.values() if v > 0)}/{len(self.fire_nodes)}")
        print(f"Total reward: {self.total_reward:.2f}")
        
        for agent_id in range(self.num_agents):
            pos = self.agent_positions[agent_id]
            water = self.agent_water_levels[agent_id]
            agent_reward = self.agent_rewards[agent_id]
            
            status = "IN TRANSIT" if self.agent_in_transit[agent_id] else "AT NODE"
            if self.agent_in_transit[agent_id]:
                source = self.agent_transit_source[agent_id]
                target = self.agent_transit_target[agent_id]
                time_left = self.agent_transit_time_remaining[agent_id]
                final_dest = self.final_destinations[agent_id]
                print(f"Agent {agent_id} - {status}: {source}->{target} Final destination: {final_dest} (Time left: {time_left}), "
                     f"Water: {water}, Reward: {agent_reward:.2f}")
            else:
                print(f"Agent {agent_id} - {status}: {pos}, Water: {water}, Reward: {agent_reward:.2f}")
        
        # Print edge occupancy
        occupied_edges = {edge: agents for edge, agents in self.edge_occupancy.items() if agents}
        if occupied_edges:
            print("\nEdge Occupancy:")
            for edge, agents in occupied_edges.items():
                print(f"  {edge}: Agents {agents}")
        
        print("---")
    
    def close(self):
        """Close the environment."""
        pass