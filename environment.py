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
    
    def __init__(self, graph=None, num_fires=10, num_water_sources=4, num_agents=2, 
                 max_steps=200, vehicle_types=None):
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
        if graph is None:
            self.graph = self._generate_graph(num_fires, num_water_sources)
        else:
            self.graph = graph
            
        # Extract nodes by type
        self.fire_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['tipo'] == 'incendio']
        self.water_nodes = [n for n, attr in self.graph.nodes(data=True) if attr['tipo'] == 'estanque']
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
        self.agent_positions = {}
        self.agent_water_levels = {}
        self.fires_remaining = {}  # Maps fire nodes to remaining water needed
        
        # Define action space (one per agent)
        self.action_space = spaces.Tuple([
            spaces.Discrete(len(self.all_nodes) + 2)  # Nodes + EXTINGUISH + REFILL
            for _ in range(self.num_agents)
        ])
        
        # Define observation space (one per agent)
        # State includes:
        # - Distance matrix from the agent's current position to all nodes (len(all_nodes))
        # - Active fire nodes (1 if fire is present, 0 otherwise) (len(all_nodes))
        # - Water source nodes (1 if it's a reservoir, 0 otherwise) (len(all_nodes))
        # - One-hot encoding of the agent's position (len(all_nodes))
        # - Agent's current water level (1)
        # - One-hot encoding of all other agents' positions (len(all_nodes) * (num_agents-1))
        
        self.state_dim = (
            len(self.all_nodes) +  # Distance matrix based on 'tiempo_viaje'
            len(self.all_nodes) +  # Active fire nodes
            len(self.all_nodes) +  # Water source nodes
            len(self.all_nodes) +  # One-hot encoding of agent's position
            1 +                    # Current water level
            len(self.all_nodes) * (self.num_agents - 1)  # Other agents' positions
        )
        
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
        
        # Place agents at random nodes that have paths to both fires and water
        valid_start_nodes = [
            node for node in self.all_nodes
            if any(nx.has_path(self.graph, node, fire) for fire in self.fire_nodes) and
            any(nx.has_path(self.graph, node, water) for water in self.water_nodes)
        ]
        
        for agent_id in range(self.num_agents):
            # Try to avoid placing agents at the same initial position
            available_nodes = [node for node in valid_start_nodes 
                              if node not in self.agent_positions.values()]
            
            if available_nodes:
                start_node = random.choice(available_nodes)
            else:
                start_node = random.choice(valid_start_nodes)
                
            self.agent_positions[agent_id] = start_node
            
            # Start with some water
            capacity = self.vehicle_types[agent_id]['capacity']
            self.agent_water_levels[agent_id] = capacity // 2
        
        # Get first observation for all agents
        observations = tuple(self._get_observation(agent_id) for agent_id in range(self.num_agents))
        info = self._get_info()
        
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
        
        # Get observations for all agents
        observations = tuple(self._get_observation(agent_id) for agent_id in range(self.num_agents))
        info = self._get_info()
        
        return observations, tuple(rewards), terminated, truncated, info
    
    def _process_agent_action(self, agent_id, action):
        """Process an individual agent's action and return the reward."""
        reward = 0
        
        # Get current position and node type
        current_pos = self.agent_positions[agent_id]
        current_node_type = self.graph.nodes[current_pos]['tipo']
        
        # Movement action
        if action < len(self.all_nodes):
            target_node = self.all_nodes[action]
            
            if target_node in self.graph.neighbors(current_pos):
                # If the node is directly connected, move there
                self.agent_positions[agent_id] = target_node
            else:
                # Find shortest path and move to the next node in the route
                try:
                    shortest_path = nx.shortest_path(self.graph, source=current_pos, 
                                                  target=target_node, weight='tiempo_viaje')
                    if len(shortest_path) > 1:
                        next_step = shortest_path[1]  # The next node in the route
                        self.agent_positions[agent_id] = next_step
                    else:
                        reward -= 100.0  # Penalize unnecessary movements
                        print(f"Agent {agent_id} attempted to move to the same node")
                except nx.NetworkXNoPath:
                    reward -= 100.0  # Penalize attempting to move to unreachable nodes (Can not happen)
        
        # Extinguish fire action
        elif action == len(self.all_nodes):
            if current_node_type == 'incendio' and self.fires_remaining[current_pos] > 0:
                water_needed = self.fires_remaining[current_pos]
                water_available = self.agent_water_levels[agent_id]
                water_width = self.vehicle_types[agent_id]['width']
                
                # Consider vehicle width for water application effectiveness
                water_used = min(water_needed, water_available)
                self.fires_remaining[current_pos] -= water_used
                self.agent_water_levels[agent_id] -= water_used
                
                reward += water_used * 0.5  # Reward for each unit of water used
                
                if self.fires_remaining[current_pos] <= 0:
                    reward += 300.0  # Extra reward for extinguishing a fire
                    self.fires_extinguished += 1
        
        # Refill water action
        elif action == len(self.all_nodes) + 1:
            if current_node_type == 'estanque':
                capacity = self.vehicle_types[agent_id]['capacity']
                # Consider refill time based on capacity
                self.agent_water_levels[agent_id] = capacity
        
        return reward
    
    def _get_observation(self, agent_id):
        """
        Generate the observation for a specific agent.
        
        Args:
            agent_id: ID of the agent to generate observation for
            
        Returns:
            numpy.ndarray: Observation vector
        """
        # Distance matrix based on 'tiempo_viaje' (normalized)
        current_pos = self.agent_positions[agent_id]
        distances = nx.single_source_dijkstra_path_length(self.graph, current_pos, weight='tiempo_viaje')
        max_dist = max(1, max(distances.values(), default=1))  # Avoid division by zero
        
        distance_matrix = np.array([
            distances.get(node, max_dist) / max_dist  # Normalize by the maximum possible distance
            for node in self.all_nodes
        ], dtype=np.float32)
        
        # Active fire nodes (1 if there's fire, 0 if extinguished)
        fire_status = np.array([
            1 if node in self.fires_remaining and self.fires_remaining[node] > 0 else 0 
            for node in self.all_nodes
        ], dtype=np.float32)
        
        # Water source nodes (1 if it's a reservoir, 0 if not)
        water_nodes = np.array([1 if node in self.water_nodes else 0 for node in self.all_nodes], dtype=np.float32)
        
        # Agent's position (one-hot encoding)
        agent_pos_encoding = np.zeros(len(self.all_nodes), dtype=np.float32)
        pos_idx = self.node_to_idx[self.agent_positions[agent_id]]
        agent_pos_encoding[pos_idx] = 1.0
        
        # Agent's water level (normalized)
        max_capacity = self.vehicle_types[agent_id]['capacity']
        water_level = np.array([self.agent_water_levels[agent_id] / max_capacity], dtype=np.float32)
        
        # Other agents' positions (one-hot encoding for each)
        other_agents_pos = np.zeros(len(self.all_nodes) * (self.num_agents - 1), dtype=np.float32)
        other_idx = 0
        
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                other_pos = self.agent_positions[other_id]
                other_pos_idx = self.node_to_idx[other_pos]
                other_agents_pos[other_idx * len(self.all_nodes) + other_pos_idx] = 1.0
                other_idx += 1
        
        # Concatenate all features into a single observation vector
        observation = np.concatenate([
            distance_matrix,    # Distance matrix based on travel time
            fire_status,        # Active fires
            water_nodes,        # Water sources
            agent_pos_encoding, # Agent's position
            water_level,        # Agent's water level
            other_agents_pos    # Other agents' positions
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
            'agent_water_levels': self.agent_water_levels
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
            print(f"Agent {agent_id} - Position: {pos}, Water: {water}, Reward: {agent_reward:.2f}")
        
        print("---")
    
    def close(self):
        """Close the environment."""
        pass
    
    def _generate_graph(self, num_fires, num_water_sources):
        """Generate a random graph for the environment."""
        G = nx.Graph()
        
        # Create nodes
        fire_nodes = [f"incendio_{i}" for i in range(num_fires)]
        water_nodes = [f"estanque_{i}" for i in range(num_water_sources)]
        nodes = fire_nodes + water_nodes
        
        # Add nodes with attributes
        for node in nodes:
            if "incendio" in node:
                G.add_node(node, tipo='incendio', water_to_extinguish=random.randint(50, 100))
            else:
                G.add_node(node, tipo='estanque', water_capacity=random.randint(1000, 2000))
        
        # Create a base connected structure
        # First, connect all nodes in a chain to ensure connectivity
        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i + 1], 
                      ancho=random.choice([2, 3]),  # Avoid very narrow paths
                      tiempo_viaje=random.randint(1, 5))
        
        # Ensure each fire has at least one path to a water source
        for fire in fire_nodes:
            if not any(nx.has_path(G, fire, water) for water in water_nodes):
                water = random.choice(water_nodes)
                G.add_edge(fire, water, 
                          ancho=random.choice([2, 3]),
                          tiempo_viaje=random.randint(1, 5))
        
        # Add additional connections for more connectivity
        for _ in range(len(nodes)):
            node1 = random.choice(nodes)
            node2 = random.choice(nodes)
            if node1 != node2 and not G.has_edge(node1, node2):
                G.add_edge(node1, node2, 
                          ancho=random.choice([2, 3]),
                          tiempo_viaje=random.randint(1, 5))
        
        return G