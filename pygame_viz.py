import pygame
import networkx as nx
import numpy as np
import math
import os
from environment import FirefightingEnv
from agent import DQNAgent
from exact_graph import generate_exact_graph

class FirefightingViz:
    def __init__(self, env, screen_width=1200, screen_height=800):
        # Inicializar PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Simulación de Lucha contra Incendios')
        
        # Almacenar referencia al entorno
        self.env = env
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Colores
        self.BACKGROUND = (240, 240, 240)
        self.NODE_COLOR = (150, 150, 150)
        self.EDGE_COLOR = (100, 100, 100)
        self.FIRE_COLOR = [(255, 200, 0), (255, 150, 0), (255, 100, 0), (255, 50, 0)]
        self.DEPOT_COLOR = (0, 150, 255)
        self.STARTER_COLOR = (0, 200, 0)
        self.AGENT_COLORS = {
            'small': (0, 255, 0),    # Verde para vehículos pequeños
            'medium': (0, 0, 255),   # Azul para vehículos medianos
            'large': (255, 0, 0)     # Rojo para vehículos grandes
        }
        self.TEXT_COLOR = (0, 0, 0)
        
        # Tamaños de nodos y aristas
        self.NODE_RADIUS = 20
        self.EDGE_WIDTH = 3
        
        # Fuentes
        self.font = pygame.font.SysFont('Arial', 12)
        self.font_large = pygame.font.SysFont('Arial', 18)
        
        # Calcular posiciones de los nodos usando el layout spring
        self.pos = nx.spring_layout(self.env.graph, seed=42)
        self.scale_and_center_layout()
        
        # Control de velocidad de los frames
        self.clock = pygame.time.Clock()
        
    def scale_and_center_layout(self):
        """Escalar y centrar el grafo para que encaje en la pantalla"""
        # Encontrar coordenadas mínimas y máximas
        min_x = min(pos[0] for pos in self.pos.values())
        max_x = max(pos[0] for pos in self.pos.values())
        min_y = min(pos[1] for pos in self.pos.values())
        max_y = max(pos[1] for pos in self.pos.values())
        
        # Calcular factores de escala con margen
        margin = 100
        scale_x = (self.screen_width - 250 - 2 * margin) / (max_x - min_x) if max_x > min_x else 1
        scale_y = (self.screen_height - 2 * margin) / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y)
        
        # Escalar y centrar
        for node in self.pos:
            x = (self.pos[node][0] - min_x) * scale + margin
            y = (self.pos[node][1] - min_y) * scale + margin
            self.pos[node] = np.array([x, y])
    
    def draw_edge(self, u, v):
        """Dibujar una arista entre los nodos u y v"""
        start_pos = self.pos[u]
        end_pos = self.pos[v]
        
        # Ajustar puntos finales para estar en la circunferencia de los nodos
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = max(1, math.sqrt(dx*dx + dy*dy))
        
        # Normalizar vector de dirección
        dx /= length
        dy /= length
        
        # Ajustar puntos finales
        start_pos_adj = (start_pos[0] + dx * self.NODE_RADIUS, 
                         start_pos[1] + dy * self.NODE_RADIUS)
        end_pos_adj = (end_pos[0] - dx * self.NODE_RADIUS, 
                       end_pos[1] - dy * self.NODE_RADIUS)
        
        # Dibujar arista
        edge_color = self.EDGE_COLOR
        # Verificar si hay agentes en la arista
        if (u, v) in self.env.edge_occupancy and self.env.edge_occupancy[(u, v)]:
            edge_color = (255, 100, 100)  # Rojo para aristas ocupadas
            
        pygame.draw.line(self.screen, edge_color, start_pos_adj, end_pos_adj, self.EDGE_WIDTH)
        
        # Dibujar información de la arista
        edge_data = self.env.graph[u][v]
        edge_width = edge_data.get('width', 1)
        midpoint = ((start_pos_adj[0] + end_pos_adj[0]) / 2,
                    (start_pos_adj[1] + end_pos_adj[1]) / 2)
        
        # Texto de ancho
        width_text = self.font.render(f"W:{edge_width}", True, self.TEXT_COLOR)
        self.screen.blit(width_text, (midpoint[0] + 5, midpoint[1] - 15))
        
        # Tiempo de tránsito
        time_text = self.font.render(f"T:{edge_data.get('transit_time', 1)}", True, self.TEXT_COLOR)
        self.screen.blit(time_text, (midpoint[0] + 5, midpoint[1] + 5))
    
    def draw_node(self, node):
        """Dibujar un nodo con su apariencia específica según el tipo"""
        x, y = self.pos[node]
        node_attrs = self.env.graph.nodes[node]
        node_type = node_attrs.get('type', 'normal')
        
        # Dibujar diferentes tipos de nodos
        if node_type == 'fire':
            # Nodo de incendio con intensidad basada en nivel actual
            if node in self.env.fire_levels:
                fire_level = self.env.fire_levels[node]
                max_fire = self.env.max_fire
                intensity = min(3, int(3 * fire_level / max_fire)) if max_fire > 0 else 0
                color = self.FIRE_COLOR[intensity]
                
                # Dibujar nodo de incendio con nivel actual
                pygame.draw.circle(self.screen, color, (int(x), int(y)), self.NODE_RADIUS)
                
                # Dibujar nivel de incendio
                level_text = self.font.render(f"{fire_level}", True, (0, 0, 0))
                self.screen.blit(level_text, (x - level_text.get_width()/2, y - level_text.get_height()/2))
            else:
                # Incendio extinguido
                pygame.draw.circle(self.screen, (100, 100, 100), (int(x), int(y)), self.NODE_RADIUS)
        
        elif node_type == 'depot':
            # Nodo depósito
            pygame.draw.circle(self.screen, self.DEPOT_COLOR, (int(x), int(y)), self.NODE_RADIUS)
            water_text = self.font.render("DEPOT", True, (0, 0, 0))
            self.screen.blit(water_text, (x - water_text.get_width()/2, y - water_text.get_height()/2))
        
        elif node_type == 'starter':
            # Nodo inicial
            pygame.draw.circle(self.screen, self.STARTER_COLOR, (int(x), int(y)), self.NODE_RADIUS)
            start_text = self.font.render("START", True, (0, 0, 0))
            self.screen.blit(start_text, (x - start_text.get_width()/2, y - start_text.get_height()/2))
        
        else:
            # Nodo normal
            pygame.draw.circle(self.screen, self.NODE_COLOR, (int(x), int(y)), self.NODE_RADIUS)
        
        # Dibujar etiqueta del nodo
        label_text = self.font.render(str(node), True, self.TEXT_COLOR)
        self.screen.blit(label_text, (x - label_text.get_width()/2, y - self.NODE_RADIUS - 20))
    
    def draw_agent(self, agent_id):
        """Dibujar un agente en la pantalla"""
        # Obtener información del agente
        vehicle_type = self.env.agent_vehicle_types[agent_id]
        water_level = self.env.agents_water[agent_id]
        max_water = self.env.agent_max_water[agent_id]
        vehicle_width = self.env.vehicle_types[vehicle_type]['width']
        agent_color = self.AGENT_COLORS[vehicle_type]
        
        # Si el agente está en tránsito, dibujar en la arista
        if self.env.agent_in_transit[agent_id]:
            source = self.env.agent_transit_source[agent_id]
            target = self.env.agent_transit_target[agent_id]
            
            if source is not None and target is not None:
                # Calcular posición a lo largo de la arista según tiempo restante
                total_time = self.env.graph[source][target]['transit_time']
                time_remaining = self.env.agent_transit_time_remaining[agent_id]
                fraction = 1 - (time_remaining / total_time) if total_time > 0 else 0
                
                source_pos = self.pos[source]
                target_pos = self.pos[target]
                
                # Interpolar posición
                agent_x = source_pos[0] + fraction * (target_pos[0] - source_pos[0])
                agent_y = source_pos[1] + fraction * (target_pos[1] - source_pos[1])
                
                # Dibujar agente como un círculo con tamaño según el tipo de vehículo
                agent_radius = 10 + vehicle_width * 3
                pygame.draw.circle(self.screen, agent_color, (int(agent_x), int(agent_y)), agent_radius)
                
                # Dibujar ID del agente
                agent_text = self.font.render(f"{agent_id}", True, (255, 255, 255))
                self.screen.blit(agent_text, (agent_x - agent_text.get_width()/2, agent_y - agent_text.get_height()/2))
                
                # Dibujar nivel de agua
                water_text = self.font.render(f"W:{water_level}/{max_water}", True, self.TEXT_COLOR)
                self.screen.blit(water_text, (agent_x - water_text.get_width()/2, agent_y - agent_radius - 20))
        else:
            # Dibujar agente en su posición actual
            position = self.env.agent_positions[agent_id]
            x, y = self.pos[position]
            
            # Desplazar el agente del centro del nodo para evitar solapamiento
            offset_angle = agent_id * (2 * math.pi / self.env.num_agents)
            offset_x = 30 * math.cos(offset_angle)
            offset_y = 30 * math.sin(offset_angle)
            
            # Dibujar agente como un círculo con tamaño según el tipo de vehículo
            agent_radius = 10 + vehicle_width * 3
            pygame.draw.circle(self.screen, agent_color, (int(x + offset_x), int(y + offset_y)), agent_radius)
            
            # Dibujar ID del agente
            agent_text = self.font.render(f"{agent_id}", True, (255, 255, 255))
            self.screen.blit(agent_text, (x + offset_x - agent_text.get_width()/2, 
                                         y + offset_y - agent_text.get_height()/2))
            
            # Dibujar nivel de agua
            water_text = self.font.render(f"W:{water_level}/{max_water}", True, self.TEXT_COLOR)
            self.screen.blit(water_text, (x + offset_x - water_text.get_width()/2, 
                                         y + offset_y - agent_radius - 20))
    
    def draw_legend(self):
        """Dibujar una leyenda para la visualización"""
        # Configurar posición y dimensiones de la leyenda
        legend_x = 10
        legend_y = 10
        line_height = 25
        
        # Dibujar fondo de la leyenda
        pygame.draw.rect(self.screen, (255, 255, 255), 
                         (legend_x, legend_y, 200, 200))
        pygame.draw.rect(self.screen, (0, 0, 0), 
                         (legend_x, legend_y, 200, 200), 1)
        
        # Título
        title_text = self.font_large.render("Leyenda", True, self.TEXT_COLOR)
        self.screen.blit(title_text, (legend_x + 10, legend_y + 10))
        
        # Tipos de nodos
        y_pos = legend_y + 40
        
        # Nodo de incendio
        pygame.draw.circle(self.screen, self.FIRE_COLOR[2], 
                         (legend_x + 15, y_pos), 10)
        node_text = self.font.render("Nodo de Incendio", True, self.TEXT_COLOR)
        self.screen.blit(node_text, (legend_x + 30, y_pos - 7))
        y_pos += line_height
        
        # Nodo depósito
        pygame.draw.circle(self.screen, self.DEPOT_COLOR, 
                         (legend_x + 15, y_pos), 10)
        node_text = self.font.render("Depósito", True, self.TEXT_COLOR)
        self.screen.blit(node_text, (legend_x + 30, y_pos - 7))
        y_pos += line_height
        
        # Nodo inicial
        pygame.draw.circle(self.screen, self.STARTER_COLOR, 
                         (legend_x + 15, y_pos), 10)
        node_text = self.font.render("Punto de Inicio", True, self.TEXT_COLOR)
        self.screen.blit(node_text, (legend_x + 30, y_pos - 7))
        y_pos += line_height
        
        # Tipos de vehículos
        for vehicle_type, color in self.AGENT_COLORS.items():
            pygame.draw.circle(self.screen, color, 
                             (legend_x + 15, y_pos), 10)
            vehicle_text = self.font.render(f"Vehículo {vehicle_type}", 
                                          True, self.TEXT_COLOR)
            self.screen.blit(vehicle_text, (legend_x + 30, y_pos - 7))
            y_pos += line_height
    
    def draw_status_panel(self):
        """Dibujar un panel con estado actual de la simulación"""
        panel_width = 250
        panel_height = self.screen_height
        panel_x = self.screen_width - panel_width
        panel_y = 0
        
        # Dibujar fondo del panel
        pygame.draw.rect(self.screen, (240, 240, 240), 
                         (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.line(self.screen, (200, 200, 200), 
                         (panel_x, 0), (panel_x, self.screen_height), 2)
        
        # Dibujar título
        title_text = self.font_large.render("Estado de la Simulación", True, self.TEXT_COLOR)
        self.screen.blit(title_text, (panel_x + 10, panel_y + 10))
        
        # Dibujar estado de incendios
        y_pos = panel_y + 50
        fires_title = self.font_large.render("Incendios", True, self.TEXT_COLOR)
        self.screen.blit(fires_title, (panel_x + 10, y_pos))
        y_pos += 30
        
        for node in self.env.fire_nodes:
            if node in self.env.fire_levels:
                fire_level = self.env.fire_levels[node]
                max_fire = self.env.max_fire
                status = "Activo" if fire_level > 0 else "Extinguido"
                color = (255, 0, 0) if fire_level > 0 else (0, 150, 0)
                
                fire_text = self.font.render(f"Nodo {node}: {fire_level}/{max_fire} - {status}", 
                                            True, color)
                self.screen.blit(fire_text, (panel_x + 20, y_pos))
                y_pos += 20
        
        # Dibujar estado de agentes
        y_pos += 20
        agents_title = self.font_large.render("Agentes", True, self.TEXT_COLOR)
        self.screen.blit(agents_title, (panel_x + 10, y_pos))
        y_pos += 30
        
        for agent_id in range(self.env.num_agents):
            # Información del agente
            vehicle_type = self.env.agent_vehicle_types[agent_id]
            water = self.env.agents_water[agent_id]
            max_water = self.env.agent_max_water[agent_id]
            position = self.env.agent_positions[agent_id]
            
            # Dibujar información
            agent_text = self.font.render(f"Agente {agent_id} ({vehicle_type}):", 
                                         True, self.TEXT_COLOR)
            self.screen.blit(agent_text, (panel_x + 20, y_pos))
            y_pos += 20
            
            # Nivel de agua
            water_text = self.font.render(f"  Agua: {water}/{max_water}", 
                                         True, self.TEXT_COLOR)
            self.screen.blit(water_text, (panel_x + 20, y_pos))
            y_pos += 20
            
            # Posición/tránsito
            if self.env.agent_in_transit[agent_id]:
                from_node = self.env.agent_transit_source[agent_id]
                to_node = self.env.agent_transit_target[agent_id]
                dest_node = self.env.final_destinations[agent_id]
                time_left = self.env.agent_transit_time_remaining[agent_id]
                
                transit_text = self.font.render(f"  {from_node} → {to_node} (→ {dest_node})", 
                                              True, self.TEXT_COLOR)
                self.screen.blit(transit_text, (panel_x + 20, y_pos))
                y_pos += 20
                
                time_text = self.font.render(f"  Tiempo restante: {time_left}", 
                                           True, self.TEXT_COLOR)
                self.screen.blit(time_text, (panel_x + 20, y_pos))
                y_pos += 20
            else:
                pos_text = self.font.render(f"  En nodo: {position}", 
                                          True, self.TEXT_COLOR)
                self.screen.blit(pos_text, (panel_x + 20, y_pos))
                y_pos += 20
            
            # Añadir espacio entre agentes
            y_pos += 10
    
    def draw(self):
        """Dibujar toda la visualización de la simulación"""
        # Limpiar pantalla
        self.screen.fill(self.BACKGROUND)
        
        # Dibujar aristas
        for u, v in self.env.graph.edges():
            self.draw_edge(u, v)
        
        # Dibujar nodos
        for node in self.env.graph.nodes():
            self.draw_node(node)
        
        # Dibujar agentes
        for agent_id in range(self.env.num_agents):
            self.draw_agent(agent_id)
        
        # Dibujar leyenda y panel de estado
        self.draw_legend()
        self.draw_status_panel()
        
        # Actualizar pantalla
        pygame.display.flip()
    
    def handle_events(self):
        """Manejar eventos de PyGame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def run_visualization(self, action_getter, max_steps=1000, fps=5):
        """Ejecutar la visualización con la función de obtención de acciones dada"""
        step = 0
        running = True
        
        while running and step < max_steps:
            # Manejar eventos
            running = self.handle_events()
            if not running:
                break
            
            # Dibujar estado actual
            self.draw()
            
            # Obtener acciones
            actions = action_getter(self.env._get_obs())
            
            # Avanzar el entorno
            _, reward, done, _ = self.env.step(actions)
            
            # Mostrar recompensa
            print(f"Step {step}, Reward: {reward}")
            
            # Verificar si el episodio ha terminado
            if done:
                print("¡Todos los incendios extinguidos!")
                break
            
            # Incrementar paso
            step += 1
            
            # Controlar velocidad de los frames
            self.clock.tick(fps)
        
        # Mantener ventana abierta al final hasta que el usuario la cierre
        end_screen = True
        while end_screen:
            end_screen = self.handle_events()
            self.draw()
            self.clock.tick(5)
        
        pygame.quit()

class InteractiveFirefightingViz(FirefightingViz):
    """Versión interactiva de la visualización con botones para controlar la simulación."""
    
    def __init__(self, env, screen_width=1200, screen_height=800):
        super().__init__(env, screen_width, screen_height)
        
        # Dimensiones y colores de los botones
        self.button_width = 120
        self.button_height = 40
        self.button_margin = 20
        self.button_color = (100, 200, 100)
        self.button_hover_color = (120, 220, 120)
        self.button_active_color = (80, 180, 80)
        self.button_text_color = (0, 0, 0)
        
        # Definir botones: [x, y, width, height, text, action]
        self.buttons = [
            [20, self.screen_height - 120, self.button_width, self.button_height, "Paso", "step"],
            [20 + self.button_width + self.button_margin, self.screen_height - 120, self.button_width, self.button_height, "Auto (on/off)", "auto"],
            [20 + 2 * (self.button_width + self.button_margin), self.screen_height - 120, self.button_width, self.button_height, "Reiniciar", "reset"],
            [20 + 3 * (self.button_width + self.button_margin), self.screen_height - 120, self.button_width, self.button_height, "Salir", "exit"]
        ]
        
        # Estado del botón automático
        self.auto_mode = False
        self.fps_auto = 2  # Velocidad predeterminada en modo automático
        
        # Estado del botón que está siendo presionado
        self.active_button = None
        
    def draw_buttons(self):
        """Dibujar los botones en la pantalla."""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = pygame.mouse.get_pressed()[0]  # Botón izquierdo
        
        for button in self.buttons:
            x, y, width, height, text, action = button
            
            # Verificar si el ratón está sobre el botón
            button_rect = pygame.Rect(x, y, width, height)
            hover = button_rect.collidepoint(mouse_pos)
            
            # Dibujar el botón con el color apropiado
            if action == "auto" and self.auto_mode:
                # Botón de auto destacado cuando está activo
                color = self.button_active_color
            elif hover:
                color = self.button_hover_color
            else:
                color = self.button_color
                
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), button_rect, 2)  # Borde
            
            # Dibujar texto
            text_surf = self.font_large.render(text, True, self.button_text_color)
            text_rect = text_surf.get_rect(center=(x + width/2, y + height/2))
            self.screen.blit(text_surf, text_rect)
            
    def draw_status_info(self, step, reward, done):
        """Dibujar información de estado sobre la simulación."""
        status_rect = pygame.Rect(20, self.screen_height - 60, 500, 40)
        pygame.draw.rect(self.screen, (240, 240, 240), status_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), status_rect, 1)
        
        # Información de estado
        status_text = f"Paso: {step} | Recompensa: {reward:.2f}"
        if done:
            status_text += " | Estado: Completado"
        else:
            status_text += " | Estado: En progreso"
            
        text_surf = self.font_large.render(status_text, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (30, self.screen_height - 50))
        
    def draw(self, step=0, reward=0, done=False):
        """Dibujar toda la visualización incluyendo los controles interactivos."""
        super().draw()  # Llamar al método de la clase padre
        
        # Dibujar botones e información de estado
        self.draw_buttons()
        self.draw_status_info(step, reward, done)
        
        # Actualizar pantalla
        pygame.display.flip()
        
    def handle_events(self):
        """Manejar eventos de PyGame y botones."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return {"action": "exit"}
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return {"action": "exit"}
                elif event.key == pygame.K_SPACE:
                    return {"action": "step"}
                elif event.key == pygame.K_a:
                    self.auto_mode = not self.auto_mode
                    return {"action": "auto_toggle"}
                elif event.key == pygame.K_r:
                    return {"action": "reset"}
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Verificar si se ha hecho clic en algún botón
                mouse_pos = pygame.mouse.get_pos()
                for button in self.buttons:
                    x, y, width, height, _, action = button
                    button_rect = pygame.Rect(x, y, width, height)
                    
                    if button_rect.collidepoint(mouse_pos):
                        if action == "step":
                            return {"action": "step"}
                        elif action == "auto":
                            self.auto_mode = not self.auto_mode
                            return {"action": "auto_toggle"}
                        elif action == "reset":
                            return {"action": "reset"}
                        elif action == "exit":
                            return {"action": "exit"}
        
        # En modo automático, simular un evento de "paso" automáticamente
        if self.auto_mode:
            return {"action": "step"}
            
        # Ninguna acción específica
        return {"action": "none"}
    
    def run_interactive_visualization(self, action_getter, max_steps=1000):
        """Ejecutar la visualización interactiva con botones."""
        step = 0
        running = True
        cumulative_reward = 0
        done = False
        
        # Inicializar estado
        state = self.env.reset()
        
        while running and step < max_steps:
            # Dibujar estado actual
            self.draw(step, cumulative_reward, done)
            
            # Esperar eventos/acciones del usuario
            event_result = self.handle_events()
            
            # Procesar acción
            if event_result["action"] == "exit":
                running = False
            elif event_result["action"] == "reset":
                step = 0
                cumulative_reward = 0
                done = False
                state = self.env.reset()
                print("Simulación reiniciada")
            elif event_result["action"] == "step" and not done:
                # Obtener acciones del agente
                actions = action_getter(state)
                
                # Avanzar el entorno
                state, reward, done, _ = self.env.step(actions)
                
                # Actualizar estadísticas
                cumulative_reward += reward
                step += 1
                
                print(f"Paso {step}, Recompensa: {reward:.2f}, Acumulada: {cumulative_reward:.2f}")
                
                if done:
                    print("¡Todos los incendios extinguidos!")
            
            # Controlar velocidad en modo automático
            if self.auto_mode:
                self.clock.tick(self.fps_auto)
            else:
                self.clock.tick(30)  # FPS para la interfaz cuando no está en modo automático
        
        pygame.quit()

def visualize_trained_agent(agent, env, steps_or_dir=1000, fps=5):
    """Visualizar un agente entrenado en el entorno"""
    def action_getter(state):
        """Obtener acciones del agente basadas en el estado"""
        return agent.select_action(state)

    # Ensure max_steps is an integer - handle case where a directory is passed
    if isinstance(steps_or_dir, str) and '/' in steps_or_dir:
        # If it looks like a directory path, use default steps
        max_steps = 1000
    else:
        # Otherwise try to convert to int
        max_steps = int(steps_or_dir)

    # Crear y ejecutar visualización
    viz = FirefightingViz(env)
    viz.run_visualization(action_getter, max_steps=max_steps, fps=fps)

def visualize_trained_agent_interactive(agent, env, max_steps=1000):
    """Visualizar un agente entrenado en el entorno con controles interactivos"""
    def action_getter(state):
        """Obtener acciones del agente basadas en el estado"""
        return agent.select_action(state)

    # Asegurar que max_steps es un entero
    max_steps = int(max_steps)

    # Crear y ejecutar visualización interactiva
    viz = InteractiveFirefightingViz(env)
    viz.run_interactive_visualization(action_getter, max_steps=max_steps)

def visualize_random_agent(env, steps=1000, fps=5):
    """Visualizar acciones aleatorias en el entorno"""
    def action_getter(_):
        """Obtener acciones aleatorias"""
        return [np.random.randint(0, len(env.nodes)) for _ in range(env.num_agents)]
    
    # Crear y ejecutar visualización
    viz = FirefightingViz(env)
    viz.run_visualization(action_getter, max_steps=steps, fps=fps)

def visualize_interactive(env, agents, steps=1000, fps=5):
    """Ejecutar visualización con agentes entrenados"""
    # Si solo se proporciona un agente, usarlo para todos
    if not isinstance(agents, list):
        agents = [agents]
    
    def action_getter(state):
        """Obtener acciones de los agentes"""
        # Si hay suficientes agentes, usar uno diferente por agente
        if len(agents) >= env.num_agents:
            actions = []
            for i in range(env.num_agents):
                # Cada agente toma su propia acción
                agent_action = agents[i].select_action(state)
                actions.append(agent_action[i])
            return actions
        else:
            # Usar el primer agente para todos
            return agents[0].select_action(state)
    
    # Crear y ejecutar visualización
    viz = FirefightingViz(env)
    viz.run_visualization(action_getter, max_steps=steps, fps=fps)

if __name__ == "__main__":
    # Crear entorno
    from exact_graph import generate_exact_graph
    
    graph = generate_exact_graph()
    env = FirefightingEnv(num_agents=3, graph=graph)
    
    # Visualizar con acciones aleatorias
    print("Iniciando visualización con acciones aleatorias...")
    visualize_random_agent(env)