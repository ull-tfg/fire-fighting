import numpy as np
import random

class MultiAgentFirefightingEnv:
    def __init__(self, graph, num_agents, vehicle_types):
        self.graph = graph
        self.num_agents = num_agents
        self.vehicle_types = vehicle_types
        self.agent_positions = {}
        self.agent_destinations = {}
        self.agent_times_remaining = {}
        self.agent_water_levels = {}
        self.agent_vehicle_types = {i: random.choice(list(vehicle_types.keys())) for i in range(num_agents)}

    def reset(self):
        # Resetear las posiciones y otros estados de los agentes
        self.agent_positions = {agent: random.choice(list(self.graph.nodes)) for agent in range(self.num_agents)}
        self.agent_destinations = {agent: None for agent in range(self.num_agents)}
        self.agent_times_remaining = {agent: 0 for agent in range(self.num_agents)}
        self.agent_water_levels = {agent: self.vehicle_types[self.agent_vehicle_types[agent]]['max_water_capacity'] 
                                  for agent in range(self.num_agents)}
        
        # Inicializar los incendios con cantidades aleatorias de agua necesaria
        for node in self.graph.nodes():
            if self.graph.nodes[node]['tipo'] == 'incendio':
                # Cantidad aleatoria entre 100 y 300 unidades de agua
                self.graph.nodes[node]['water_to_extinguish'] = random.uniform(100, 300)
        
        return self.get_state()
    
    def get_state(self):
        state = []
        for agent in range(self.num_agents):
            position = self.agent_positions[agent]
            destination = self.agent_destinations[agent]
            vehicle_type = self.agent_vehicle_types[agent]

            # Información básica del agente
            position_index = list(self.graph.nodes).index(position)
            destination_index = list(self.graph.nodes).index(destination) if destination else -1
            water_level = self.agent_water_levels[agent]
            max_water = self.vehicle_types[vehicle_type]['max_water_capacity']
        
            # Información del entorno cercano
            nearest_fire = float('inf')
            nearest_pond = float('inf')
            for node in self.graph.nodes():
                if self.graph.has_edge(position, node):
                    node_type = self.graph.nodes[node]['tipo']
                    if node_type == 'incendio' and self.graph.nodes[node]['water_to_extinguish'] > 0:
                        nearest_fire = min(nearest_fire, self.graph[position][node]['tiempo_viaje'])
                    elif node_type == 'estanque':
                        nearest_pond = min(nearest_pond, self.graph[position][node]['tiempo_viaje'])
            # Normalizar valores
            state.extend([
                position_index / len(self.graph.nodes),
                destination_index / len(self.graph.nodes) if destination_index >= 0 else 0,
                water_level / max_water,
                1.0 if nearest_fire == float('inf') else nearest_fire / 10,
                1.0 if nearest_pond == float('inf') else nearest_pond / 10
            ])
        return np.array(state, dtype=np.float32)
    
    def step(self, actions):
        rewards = {}
        
        for agent, action in actions.items():
            current_position = self.agent_positions[agent]
            vehicle_type = self.agent_vehicle_types[agent]
            rewards[agent] = 0
            
            # Si el agente está en movimiento, continuar el movimiento actual
            if self.agent_destinations[agent] is not None:
                self.agent_times_remaining[agent] -= 1
                if self.agent_times_remaining[agent] <= 0:
                    self.agent_positions[agent] = self.agent_destinations[agent]
                    self.agent_destinations[agent] = None
                continue
    
            # Procesar acciones según el tipo
            if action == 0:  # MOVE
                # Encontrar el mejor destino según el estado del agente
                best_destination = None
                min_distance = float('inf')
                
                for neighbor in self.graph.neighbors(current_position):
                    path = self.graph[current_position][neighbor]
                    
                    # Verificar si el vehículo puede usar el camino
                    if path['ancho'] >= self.vehicle_types[vehicle_type]['width']:
                        node_type = self.graph.nodes[neighbor]['tipo']
                        
                        # Priorizar destinos según el estado del agente
                        if self.agent_water_levels[agent] > 0:
                            # Si tiene agua, buscar incendios cercanos
                            if (node_type == 'incendio' and 
                                self.graph.nodes[neighbor]['water_to_extinguish'] > 0 and
                                path['tiempo_viaje'] < min_distance):
                                best_destination = neighbor
                                min_distance = path['tiempo_viaje']
                        else:
                            # Si no tiene agua, buscar estanques cercanos
                            if node_type == 'estanque' and path['tiempo_viaje'] < min_distance:
                                best_destination = neighbor
                                min_distance = path['tiempo_viaje']
                
                if best_destination is not None:
                    self.agent_destinations[agent] = best_destination
                    self.agent_times_remaining[agent] = self.graph[current_position][best_destination]['tiempo_viaje']
                    rewards[agent] = 0.1  # Pequeña recompensa por moverse estratégicamente
    
            elif action == 1:  # EXTINGUISH
                node_type = self.graph.nodes[current_position]['tipo']
                if node_type == 'incendio':
                    water_needed = self.graph.nodes[current_position]['water_to_extinguish']
                    if water_needed > 0 and self.agent_water_levels[agent] > 0:
                        water_used = min(
                            self.vehicle_types[vehicle_type]['water_dispense_rate'],
                            self.agent_water_levels[agent],
                            water_needed
                        )
                        self.agent_water_levels[agent] -= water_used
                        self.graph.nodes[current_position]['water_to_extinguish'] -= water_used
                        
                        # Recompensa proporcional al agua utilizada
                        rewards[agent] = 1.0 * (water_used / self.vehicle_types[vehicle_type]['water_dispense_rate'])
                        
                        # Bonus por extinguir completamente el incendio
                        if self.graph.nodes[current_position]['water_to_extinguish'] <= 0:
                            rewards[agent] += 2.0
                    else:
                        rewards[agent] = -0.1  # Penalización por intentar extinguir sin agua
                else:
                    rewards[agent] = -0.1  # Penalización por intentar extinguir en lugar incorrecto
    
            elif action == 2:  # REFILL
                node_type = self.graph.nodes[current_position]['tipo']
                if node_type == 'estanque':
                    if self.agent_water_levels[agent] < self.vehicle_types[vehicle_type]['max_water_capacity']:
                        water_added = self.vehicle_types[vehicle_type]['max_water_capacity'] - self.agent_water_levels[agent]
                        self.agent_water_levels[agent] = self.vehicle_types[vehicle_type]['max_water_capacity']
                        rewards[agent] = 0.5 * (water_added / self.vehicle_types[vehicle_type]['max_water_capacity'])
                    else:
                        rewards[agent] = -0.1  # Penalización por intentar recargar estando lleno
                else:
                    rewards[agent] = -0.1  # Penalización por intentar recargar en lugar incorrecto
    
        done = self.is_done()
        return self.get_state(), rewards, done

    def calculate_reward(self, position, agent):
        node_type = self.graph.nodes[position]['tipo']
        vehicle_type = self.agent_vehicle_types[agent]
        reward = 0

        if node_type == 'incendio':
            water_needed = self.graph.nodes[position]['water_to_extinguish']
            if water_needed > 0 and self.agent_water_levels[agent] > 0:
                water_used = min(
                    self.vehicle_types[vehicle_type]['water_dispense_rate'],
                    self.agent_water_levels[agent],
                    water_needed
                )
                # Recompensa proporcional al agua utilizada
                reward = 1.0 * (water_used / self.vehicle_types[vehicle_type]['water_dispense_rate'])
                
                self.agent_water_levels[agent] -= water_used
                self.graph.nodes[position]['water_to_extinguish'] -= water_used

                # Bonus por extinguir completamente el incendio
                if self.graph.nodes[position]['water_to_extinguish'] <= 0:
                    reward += 2.0

        elif node_type == 'estanque':
            if self.agent_water_levels[agent] < self.vehicle_types[vehicle_type]['max_water_capacity']:
                water_added = self.vehicle_types[vehicle_type]['max_water_capacity'] - self.agent_water_levels[agent]
                reward = 0.5 * (water_added / self.vehicle_types[vehicle_type]['max_water_capacity'])
                self.agent_water_levels[agent] = self.vehicle_types[vehicle_type]['max_water_capacity']

        return reward

    def is_done(self):
    # Buscar si queda algún nodo con tipo 'incendio' y water_to_extinguish > 0
        for node in self.graph.nodes:
            if (self.graph.nodes[node]['tipo'] == 'incendio' and
                self.graph.nodes[node]['water_to_extinguish'] > 0):
                return False
        return True