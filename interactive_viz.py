import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import os
from matplotlib.widgets import Button, CheckButtons
from datetime import datetime

class InteractiveGraphViz:
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.current_step = 0
        self.max_steps = 300
        self.done = False
        self.states = None
        self.info = None
        self.agent_paths = {i: [] for i in range(env.num_agents)}
        
        # Figura principal y layout
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.subplots_adjust(bottom=0.2)
        
        # Posicionamiento de nodos en el grafo (fixed por consistencia)
        self.pos = nx.spring_layout(self.env.graph, seed=42)
        
        # Añadir directorio para guardar imágenes
        self.save_dir = os.path.join(os.getcwd(), "visualizations")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Timestamp para nombrar archivos
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_count = 0
        self.auto_save = False
        
        # Posición de los controles
        self.btn_step = plt.axes([0.6, 0.05, 0.1, 0.075])
        self.btn_reset = plt.axes([0.71, 0.05, 0.1, 0.075])
        self.btn_save = plt.axes([0.82, 0.05, 0.1, 0.075])
        self.chk_auto = plt.axes([0.82, 0.13, 0.15, 0.05])
        
        # Crear botones
        self.button_step = Button(self.btn_step, 'Step')
        self.button_step.on_clicked(self.step_forward)
        
        self.button_reset = Button(self.btn_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_sim)
        
        self.button_save = Button(self.btn_save, 'Save')
        self.button_save.on_clicked(self.save_current_frame)
        
        self.check_auto = CheckButtons(self.chk_auto, ['Auto-save'], [False])
        self.check_auto.on_clicked(self.toggle_auto_save)
        
        # Inicializar el entorno
        self.reset_sim(None)
        
    def reset_sim(self, event):
        """Reinicia la simulación"""
        self.current_step = 0
        self.done = False
        self.states, self.info = self.env.reset()
        
        # Limpiar caminos de agentes
        self.agent_paths = {i: [] for i in range(self.env.num_agents)}
        for agent_id in range(self.env.num_agents):
            self.agent_paths[agent_id].append(self.env.agent_positions[agent_id])
            
        self.update_plot()
        self.fig.canvas.draw_idle()

    def toggle_auto_save(self, label):
        """Toggle automático para guardar cada paso"""
        self.auto_save = not self.auto_save
        print(f"Auto-save {'activado' if self.auto_save else 'desactivado'}")
    
    def save_current_frame(self, event=None):
        """Guarda la visualización actual como imagen"""
        # Crear nombre de archivo con timestamp y número de paso
        filename = f"{self.timestamp}_step{self.current_step:03d}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        # Guardar figura
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Imagen guardada: {filepath}")
        
        self.frame_count += 1
    
    def step_forward(self, event):
        """Avanza un paso en la simulación"""
        if self.done or self.current_step >= self.max_steps:
            return
            
        # Seleccionar acciones
        actions = []
        for agent_id, agent in enumerate(self.agents):
            state_tensor = torch.FloatTensor(self.states[agent_id]).unsqueeze(0)
            available_actions = [
                self.env.all_actions.index(action) 
                for action in self.env.agent_action_spaces[agent_id]['available']
            ]
            action = agent.select_action(state_tensor, available_actions)
            actions.append(action.item())
            
        # Ejecutar un paso en el entorno
        self.states, rewards, terminated, truncated, self.info = self.env.step(actions)
        
        # Actualizar caminos de los agentes
        for agent_id in range(self.env.num_agents):
            if not self.env.agent_in_transit[agent_id]:
                self.agent_paths[agent_id].append(self.env.agent_positions[agent_id])
                
        self.current_step += 1
        self.done = terminated or truncated
        
        # Actualizar visualización
        self.update_plot()
        self.fig.canvas.draw_idle()
        
        # Auto-guardar si está activado
        if self.auto_save:
            self.save_current_frame()
    
    def save_animation(self, fps=1, quality=95, optimize=True):
        """Crea un GIF con opciones avanzadas"""
        try:
            from PIL import Image
            import glob

            # Obtener lista de imágenes
            frames = sorted(glob.glob(os.path.join(self.save_dir, f"{self.timestamp}_step*.png")))

            # Abrir imágenes
            images = [Image.open(frame) for frame in frames]

            # Calcular duración basada en fps
            duration = int(1000 / fps)

            # Guardar GIF con más opciones
            images[0].save(
                os.path.join(self.save_dir, f"{self.timestamp}_animation_fps{fps}.gif"),
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
                optimize=optimize,
                quality=quality
            )

        except ImportError:
            print("Necesitas instalar Pillow: pip install Pillow")
    
    def update_plot(self):
        """Actualiza el gráfico con el estado actual"""
        self.ax.clear()
        
        # Extraer información actual
        fires_active = [n for n in self.env.fire_nodes if 
                       n in self.env.fires_remaining and self.env.fires_remaining[n] > 0]
        fires_extinguished = [n for n in self.env.fire_nodes if 
                             n not in self.env.fires_remaining or self.env.fires_remaining[n] <= 0]
        
        # Preparar colores de nodos
        node_colors = []
        for node in self.env.graph.nodes():
            if node in fires_active:
                node_colors.append('red')
            elif node in fires_extinguished:
                node_colors.append('gray')
            elif node in self.env.tank_nodes:
                node_colors.append('blue')
            else:  # starter nodes
                node_colors.append('yellow')
        
        # Dibujar grafo base
        nx.draw(self.env.graph, pos=self.pos, node_color=node_colors, 
                with_labels=True, node_size=500, alpha=0.8, ax=self.ax)
        
        # Dibujar tiempos de tránsito
        edge_labels = nx.get_edge_attributes(self.env.graph, 'transit_time')
        nx.draw_networkx_edge_labels(self.env.graph, self.pos, edge_labels=edge_labels, ax=self.ax)
        
        # Dibujar agentes en sus posiciones
        for agent_id in range(self.env.num_agents):
            if not self.env.agent_in_transit[agent_id]:
                pos = self.env.agent_positions[agent_id]
                water = self.env.agent_water_levels[agent_id]
                agent_x, agent_y = self.pos[pos]
                
                # Agente representado como un triángulo
                self.ax.scatter(agent_x, agent_y, s=300, marker='^', 
                               color=f'C{agent_id}', label=f'Agent {agent_id}', zorder=10)
                
                # Mostrar nivel de agua
                self.ax.text(agent_x, agent_y-0.05, f'W:{water:.0f}', 
                            ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
            else:
                # Agente en tránsito - mostrar en la arista
                source = self.env.agent_transit_source[agent_id]
                target = self.env.agent_transit_target[agent_id]
                src_x, src_y = self.pos[source]
                tgt_x, tgt_y = self.pos[target]
                
                # Calcular posición en función del tiempo restante
                total_time = self.env.graph[source][target]['transit_time']
                remaining = self.env.agent_transit_time_remaining[agent_id]
                progress = (total_time - remaining) / total_time
                
                # Posición interpolada
                x = src_x + progress * (tgt_x - src_x)
                y = src_y + progress * (tgt_y - src_y)
                
                # Dibujar agente en tránsito
                self.ax.scatter(x, y, s=200, marker='>', 
                               color=f'C{agent_id}', label=f'Agent {agent_id} (transit)', zorder=10)
                
                # Mostrar tiempo restante
                self.ax.text(x, y-0.05, f'T:{remaining}', 
                            ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        
        # Dibujar líneas de trayectoria de cada agente
        for agent_id, path in self.agent_paths.items():
            if len(path) > 1:
                path_x = [self.pos[node][0] for node in path]
                path_y = [self.pos[node][1] for node in path]
                self.ax.plot(path_x, path_y, '-', color=f'C{agent_id}', alpha=0.5, linewidth=1.5)
        
        # Mostrar información del estado actual
        title = (f"Step: {self.current_step}/{self.max_steps}\n"
                f"Fires Remaining: {len(fires_active)}/{len(self.env.fire_nodes)}\n"
                f"Total Reward: {self.env.total_reward:.2f}")
        self.ax.set_title(title)
        
        # Mostrar leyenda con estados de los agentes
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), 
                      loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    def show(self):
        """Muestra la visualización y registra evento de cierre"""
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show()
    
    def on_close(self, event):
        """Acciones a realizar cuando se cierra la figura"""
        if self.frame_count > 1:
            print("\n¿Quieres crear un GIF con todas las imágenes guardadas? (s/n)")
            response = input().lower()
            if response == 's' or response == 'si':
                self.save_animation()

# Para usar la visualización:
def visualize_interactive(env, agents):
    viz = InteractiveGraphViz(env, agents)
    viz.show()