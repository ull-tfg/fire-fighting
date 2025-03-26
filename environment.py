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
            1 +                      # Current water level
            len(self.all_nodes) * (self.num_agents - 1)   # Other agents' positions
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
                'available': self.all_actions.copy()  # Este es el que se actualizará dinámicamente
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
            start_node = random.choice(available_nodes)
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
        
        # Process each agent's action
        for agent_id, action in enumerate(actions):
            # ESTA ES LA LÍNEA CLAVE - Las acciones YA son índices en all_actions
            # No necesitamos convertirlas, solo verificar que sean válidas
            target_node = self.all_actions[action]
            
            # Verificar si el nodo destino es uno válido para este agente
            if target_node not in self.agent_action_spaces[agent_id]['available']:
                #print(f"Invalid target for agent {agent_id}: {target_node}, staying put.")
                rewards[agent_id] = 0
                self.agent_rewards[agent_id] += 0   # No reward for invalid action (stay put) but should penalize?
                continue
            
            reward = self._process_agent_action(agent_id, action)
            rewards[agent_id] = reward
            self.agent_rewards[agent_id] += reward
        
        # Update global reward
        self.total_reward += sum(rewards)
        
        # Advance step and check episode completion
        self.current_step += 1
        
        # Terminate if all fires are extinguished
        terminated = all(fire <= 0 for fire in self.fires_remaining.values())
        truncated = self.current_step >= self.max_steps

        # Bonus reward for extinguishing all fires
        if terminated:
            for agent_id in range(self.num_agents):
                rewards[agent_id] += 1000.0
        
        # Get observations for all agents
        observations = tuple(self._get_observation(agent_id) for agent_id in range(self.num_agents))
        info = self._get_info()
        
        return observations, tuple(rewards), terminated, truncated, info
    
    def _update_agent_action_space(self, agent_id):
        """Actualiza el espacio de acciones para un agente basado en su posición actual."""
        available_actions = []

        # PRIORIDAD 1: Incendios activos si tiene agua
        if self.agent_water_levels[agent_id] > 0:
            for fire_node in self.fire_actions:
                if (fire_node in self.fires_remaining and 
                    self.fires_remaining[fire_node] > 0):
                    available_actions.append(fire_node)

        # PRIORIDAD 2: Tanques si necesita agua
        if self.agent_water_levels[agent_id] < self.vehicle_types[agent_id]['capacity']:
            for tank_node in self.tank_actions:
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
    
    def handle_fire_action(self, agent_id, fire_node):
        """Handle an agent's action to target a fire node."""
        reward = 0

        # Check if fire still exists
        if fire_node not in self.fires_remaining or self.fires_remaining[fire_node] <= 0:
            return reward
        # Check if agent has enough water
        if self.agent_water_levels[agent_id] <= 0:
            return reward
        # Move to the fire node
        self.agent_positions[agent_id] = fire_node
        # Try to extinguish the fire
        water_needed = self.fires_remaining[fire_node]
        water_available = self.agent_water_levels[agent_id]
        water_used = min(water_needed, water_available)
        self.fires_remaining[fire_node] -= water_used
        self.agent_water_levels[agent_id] -= water_used
        # Update rewards
        reward = water_used  # Reward for water used
        if self.fires_remaining[fire_node] <= 0:
            reward += 400.0  # Bonus for extinguishing fire
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
        self.agent_positions[agent_id] = tank_node
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

        # Active fire nodes (1 if there's fire, 0 if extinguished)
        fire_status = np.array([
            1 if node in self.fires_remaining and self.fires_remaining[node] > 0 else 0 
            for node in self.all_nodes
        ], dtype=np.float32)

        # Water source nodes (1 if it's a reservoir, 0 if not)
        tank_nodes = np.array([1 if node in self.tank_nodes else 0 for node in self.all_nodes], dtype=np.float32)

        # Agent's position (one-hot encoding)
        agent_pos_encoding = np.zeros(len(self.all_nodes), dtype=np.float32)
        pos_idx = self.node_to_idx[self.agent_positions[agent_id]]
        agent_pos_encoding[pos_idx] = 1.0

        # Agent's water level (normalized)
        max_capacity = self.vehicle_types[agent_id]['capacity']
        water_level = np.array([self.agent_water_levels[agent_id] / max_capacity], dtype=np.float32)

        # Other agents' positions (one-hot encoding for each)
        other_agents_pos = np.zeros(len(self.all_nodes) * (self.num_agents - 1), dtype=np.float32)


        # Concatenate all features into a single observation vector
        observation = np.concatenate([
            adjacency_matrix,      # Full graph adjacency matrix
            fire_status,           # Active fires
            tank_nodes,           # Water sources
            agent_pos_encoding,    # Agent's position
            water_level,           # Agent's water level
            other_agents_pos       # Other agents' positions
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