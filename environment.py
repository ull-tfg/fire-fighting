import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from enum import Enum
import random
from collections import defaultdict

class Action(Enum):
    MOVE = 0
    REFILL = 1
    EXTINGUISH = 2

class FirefightingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, graph=None, num_fires=10, num_water_sources=4, num_agents=1, 
                 max_steps=200, vehicle_types=None, render_mode=None):
        """
        Initialize the Firefighting Environment.
        
        Args:
            graph: NetworkX graph representing the environment. If None, a random graph will be generated.
            num_fires: Number of fires to generate if graph is None.
            num_water_sources: Number of water sources to generate if graph is None.
            num_agents: Number of firefighting agents.
            max_steps: Maximum number of steps per episode.
            vehicle_types: Dict mapping agent_id to vehicle type properties {'capacity': int, 'width': int}.
            render_mode: 'human' or 'rgb_array' for visualization.
        """
        super().__init__()
        
        # Environment parameters
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        
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
        
        # Define action space: MOVE, REFILL, EXTINGUISH
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # State includes:
        # - One-hot encoding of agent position (len(all_nodes))
        # - Agent's current water level (1)
        # - Status of each fire (water left to extinguish) (len(fire_nodes))
        # - Water capacity of each water source (len(water_nodes))
        # - Distances to all nodes (len(all_nodes))
        self.state_dim = (
            len(self.all_nodes) +  # One-hot position
            1 +                   # Current water level
            len(self.fire_nodes) +  # Fire statuses
            len(self.water_nodes) +  # Water source capacities
            len(self.all_nodes)     # Distances
        )
        
        low = np.zeros(self.state_dim, dtype=np.float32)
        high = np.ones(self.state_dim, dtype=np.float32) * float('inf')
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Statistics tracking
        self.total_reward = 0
        self.fires_extinguished = 0
        
        # Initialize the state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_reward = 0
        self.fires_extinguished = 0
        
        # Reset fire statuses
        self.fires_remaining = {
            node: self.graph.nodes[node]['water_to_extinguish'] 
            for node in self.fire_nodes
        }
        
        # Reset agent positions and water levels
        self.agent_positions = {}
        self.agent_water_levels = {}
        
        for agent_id in range(self.num_agents):
            # Place agent at a random node that has a path to both fires and water
            valid_start_nodes = [
                node for node in self.all_nodes
                if any(nx.has_path(self.graph, node, fire) for fire in self.fire_nodes) and
                any(nx.has_path(self.graph, node, water) for water in self.water_nodes)
            ]
            start_node = random.choice(valid_start_nodes)
            self.agent_positions[agent_id] = start_node
            
            # Start with some water
            capacity = self.vehicle_types[agent_id]['capacity']
            self.agent_water_levels[agent_id] = capacity // 2
        
        # Get first observation
        observation = self._get_observation(0)  # Observation for first agent
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Integer representing the action to take (MOVE, REFILL, EXTINGUISH)
            
        Returns:
            observation: Current observation
            reward: Reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated due to time limit
            info: Additional information
        """
        # For simplicity, we'll focus on a single agent (agent_id=0)
        agent_id = 0
        reward = 0
        action_enum = Action(action)
        
        # Get current position and node type
        current_pos = self.agent_positions[agent_id]
        current_node_type = self.graph.nodes[current_pos]['tipo']
        
        # Process action
        if action_enum == Action.MOVE:
            # Get neighboring nodes that the agent can access based on vehicle width
            valid_neighbors = []
            for neighbor in self.graph.neighbors(current_pos):
                edge_width = self.graph[current_pos][neighbor]['ancho']
                if edge_width >= self.vehicle_types[agent_id]['width']:
                    valid_neighbors.append(neighbor)
            
            if valid_neighbors:
                # Move to a random valid neighbor
                new_pos = random.choice(valid_neighbors)
                travel_time = self.graph[current_pos][new_pos]['tiempo_viaje']
                
                # Apply cost for traveling
                reward -= travel_time * 0.1  # Small penalty for travel time
                self.agent_positions[agent_id] = new_pos
            else:
                # Can't move, apply small penalty
                reward -= 1.0
        
        elif action_enum == Action.REFILL:
            if current_node_type == 'estanque':
                # Refill to capacity
                capacity = self.vehicle_types[agent_id]['capacity']
                refill_amount = capacity - self.agent_water_levels[agent_id]
                self.agent_water_levels[agent_id] = capacity
                reward += 0.5  # Small reward for successful refill
            else:
                # Not at a water source, apply penalty
                reward -= 1.0
        
        elif action_enum == Action.EXTINGUISH:
            if current_node_type == 'incendio' and self.fires_remaining[current_pos] > 0:
                # Determine how much water to use
                water_needed = self.fires_remaining[current_pos]
                water_available = self.agent_water_levels[agent_id]
                water_used = min(water_needed, water_available)
                
                # Update fire and agent water
                self.fires_remaining[current_pos] -= water_used
                self.agent_water_levels[agent_id] -= water_used
                
                # Reward based on water used and if fire is extinguished
                reward += water_used * 0.2  # Reward proportional to water used
                
                if self.fires_remaining[current_pos] <= 0:
                    reward += 10.0  # Bonus for extinguishing a fire
                    self.fires_extinguished += 1
            else:
                # Not at a fire or fire already extinguished
                reward -= 1.0
        
        self.current_step += 1
        self.total_reward += reward
        
        # Check termination conditions
        terminated = all(water <= 0 for water in self.fires_remaining.values())
        truncated = self.current_step >= self.max_steps
        
        # Get observation
        observation = self._get_observation(agent_id)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self, agent_id):
        """
        Construct the observation for the given agent.
        """
        observation = np.zeros(self.state_dim, dtype=np.float32)
        
        # One-hot encoding of position
        pos_idx = self.node_to_idx[self.agent_positions[agent_id]]
        observation[pos_idx] = 1.0
        
        # Current water level (normalized by capacity)
        capacity = self.vehicle_types[agent_id]['capacity']
        current_water = self.agent_water_levels[agent_id]
        observation[len(self.all_nodes)] = current_water / capacity
        
        # Fire statuses (normalized by initial water needed)
        fire_offset = len(self.all_nodes) + 1
        for i, fire_node in enumerate(self.fire_nodes):
            initial_water = self.graph.nodes[fire_node]['water_to_extinguish']
            remaining_water = self.fires_remaining[fire_node]
            observation[fire_offset + i] = remaining_water / initial_water
        
        # Water source capacities
        water_offset = fire_offset + len(self.fire_nodes)
        for i, water_node in enumerate(self.water_nodes):
            capacity = self.graph.nodes[water_node]['water_capacity']
            observation[water_offset + i] = capacity / 2000.0  # Normalize by max capacity
        
        # Distances to all nodes (normalized by maximum distance)
        dist_offset = water_offset + len(self.water_nodes)
        current_pos = self.agent_positions[agent_id]
        
        # Calculate shortest paths from current position to all nodes
        distances = nx.single_source_shortest_path_length(self.graph, current_pos)
        max_dist = max(distances.values()) if distances else 1
        
        for node, idx in self.node_to_idx.items():
            if node in distances:
                normalized_dist = distances[node] / max_dist
                observation[dist_offset + idx] = normalized_dist
            else:
                observation[dist_offset + idx] = 1.0  # Unreachable nodes get max distance
        
        return observation
    
    def _get_info(self):
        """Return additional information about the current state."""
        return {
            'fires_remaining': sum(1 for v in self.fires_remaining.values() if v > 0),
            'fires_extinguished': self.fires_extinguished,
            'total_reward': self.total_reward,
            'steps': self.current_step
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        # Implementation would depend on desired visualization
        # For now, we'll just print some basic information
        if self.render_mode == "human":
            agent_id = 0
            pos = self.agent_positions[agent_id]
            water = self.agent_water_levels[agent_id]
            fires_left = sum(1 for v in self.fires_remaining.values() if v > 0)
            
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Agent at: {pos}, Water: {water}")
            print(f"Fires left: {fires_left}/{len(self.fire_nodes)}")
            print(f"Total reward: {self.total_reward:.2f}")
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