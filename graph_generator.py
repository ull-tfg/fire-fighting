import networkx as nx
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

def generate_variable_graphs(
    num_graphs: int = 5,
    base_nodes: int = 15,
    connectivity: str = "medium",  # "low", "medium", "high"
    num_depots: str = "medium",    # "low", "medium", "high"
    num_fires: str = "medium",     # "low", "medium", "high"
    edge_widths: str = "medium",   # "low", "medium", "high"
    transit_times: str = "medium", # "low", "medium", "high"
    seed: int = None
) -> List[nx.Graph]:
    """
    Genera una lista de grafos con características variables para probar algoritmos de extinción de incendios.
    
    Args:
        num_graphs: Número de grafos a generar (por defecto 5)
        base_nodes: Número base de nodos para cada grafo
        connectivity: Nivel de conectividad ("low", "medium", "high")
        num_depots: Cantidad de depósitos ("low", "medium", "high") 
        num_fires: Cantidad de incendios ("low", "medium", "high")
        edge_widths: Anchos de aristas por lo general ("low", "medium", "high")
        transit_times: Tiempos de tránsito por lo general ("low", "medium", "high")
        seed: Semilla para reproducibilidad
        
    Returns:
        Lista de grafos de NetworkX configurados para el entorno de extinción de incendios
    """
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Definir rangos según los parámetros
    connectivity_ranges = {
        "low": 0.15,       # Grafos poco conectados
        "medium": 0.3,     # Conectividad moderada  
        "high": 0.6        # Grafos muy conectados
    }
    
    depot_ranges = {
        "low": (1, 2),
        "medium": (2, 4),
        "high": (3, 5)
    }
    
    fire_ranges = {
        "low": (2, 4),
        "medium": (3, 6),
        "high": (4, 8)
    }
    
    width_ranges = {
        "low": (1, 3),      # Anchos pequeños
        "medium": (2, 5),   # Anchos medianos
        "high": (3, 7)      # Anchos grandes
    }
    
    time_ranges = {
        "low": (1, 3),      # Tiempos rápidos
        "medium": (2, 5),   # Tiempos moderados
        "high": (3, 7)      # Tiempos lentos
    }
    
    graphs = []
    
    for i in range(num_graphs):
        print(f"Generando grafo {i+1}/{num_graphs}...")
        
        # Variar ligeramente el número de nodos para cada grafo
        num_nodes = base_nodes + random.randint(-2, 3)
        
        # Generar grafo base con conectividad específica
        graph = _generate_base_graph(num_nodes, connectivity_ranges[connectivity])
        
        # Asignar características a las aristas
        _assign_edge_properties(graph, width_ranges[edge_widths], time_ranges[transit_times])
        
        # Asignar tipos de nodos
        _assign_node_types(graph, depot_ranges[num_depots], fire_ranges[num_fires])
        
        # Validar que el grafo sea conexo y tenga las propiedades necesarias
        if _validate_graph(graph):
            graphs.append(graph)
            print(f"  ✓ Grafo {i+1} generado exitosamente")
            print(f"    - Nodos: {len(graph.nodes())}")
            print(f"    - Aristas: {len(graph.edges())}")
            print(f"    - Depósitos: {len([n for n, d in graph.nodes(data=True) if d.get('type') == 'depot'])}")
            print(f"    - Incendios: {len([n for n, d in graph.nodes(data=True) if d.get('type') == 'fire'])}")
        else:
            print(f"  ✗ Grafo {i+1} no válido, regenerando...")
            i -= 1  # Regenerar este grafo
    
    return graphs

def _generate_base_graph(num_nodes: int, connectivity_value: float) -> nx.Graph:
    """Genera un grafo base con la conectividad especificada."""
    
    # Usar la probabilidad de conexión especificada
    p = connectivity_value
    
    # Intentar generar un grafo conexo
    max_attempts = 10
    for attempt in range(max_attempts):
        # Usar Erdős–Rényi con probabilidad ajustada
        graph = nx.erdos_renyi_graph(num_nodes, p)
        
        # Si no es conexo, agregar aristas para conectarlo
        if not nx.is_connected(graph):
            # Encontrar componentes conectados
            components = list(nx.connected_components(graph))
            
            # Conectar componentes
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                graph.add_edge(node1, node2)
        
        # Verificar que sea conexo
        if nx.is_connected(graph):
            break
    else:
        # Si no se pudo generar un grafo conexo, usar un grafo en línea como fallback
        graph = nx.path_graph(num_nodes)
        # Agregar algunas aristas aleatorias para mayor conectividad
        additional_edges = int(num_nodes * 0.2)
        for _ in range(additional_edges):
            u, v = random.sample(range(num_nodes), 2)
            graph.add_edge(u, v)
    
    return graph

def _assign_edge_properties(graph: nx.Graph, width_range: Tuple[int, int], time_range: Tuple[int, int]):
    """Asigna propiedades de ancho y tiempo de tránsito a las aristas."""
    
    for u, v in graph.edges():
        # Asignar ancho de arista
        width = random.randint(width_range[0], width_range[1])
        
        # Asignar tiempo de tránsito (inversamente relacionado con el ancho)
        # Aristas más anchas tienden a ser más rápidas
        base_time = random.randint(time_range[0], time_range[1])
        time_modifier = max(0.5, 1.5 - (width / max(width_range)))
        transit_time = max(1, int(base_time * time_modifier))
        
        graph[u][v]['width'] = width
        graph[u][v]['transit_time'] = transit_time

def _assign_node_types(graph: nx.Graph, depot_range: Tuple[int, int], fire_range: Tuple[int, int]):
    """Asigna tipos a los nodos del grafo. Solo pueden ser starter, depot o fire."""
    
    nodes = list(graph.nodes())
    num_nodes = len(nodes)
    
    # Siempre debe haber al menos 1 starter, 1 depot y 1 fire
    min_starter = 1
    min_depot = max(1, depot_range[0])
    min_fire = max(1, fire_range[0])
    
    # Verificar que hay suficientes nodos para los mínimos requeridos
    if min_starter + min_depot + min_fire > num_nodes:
        # Ajustar a los mínimos absolutos
        min_depot = 1
        min_fire = 1
        
        if min_starter + min_depot + min_fire > num_nodes:
            raise ValueError(f"No hay suficientes nodos ({num_nodes}) para los tipos mínimos requeridos")
    
    # Calcular cantidades máximas
    max_depot = min(depot_range[1], num_nodes - min_starter - min_fire)
    max_fire = min(fire_range[1], num_nodes - min_starter - min_depot)
    
    # Determinar cantidades finales
    num_depots = random.randint(min_depot, max_depot)
    num_fires = random.randint(min_fire, max_fire)
    
    # Los nodos restantes serán starters adicionales
    num_starters = num_nodes - num_depots - num_fires
    
    # Crear lista de tipos para asignar
    node_types = ['starter'] * num_starters + ['depot'] * num_depots + ['fire'] * num_fires
    
    # Mezclar aleatoriamente para distribución aleatoria
    random.shuffle(node_types)
    
    # Asignar tipos a los nodos
    for node, node_type in zip(nodes, node_types):
        graph.nodes[node]['type'] = node_type

def _validate_graph(graph: nx.Graph) -> bool:
    """Valida que el grafo tenga las propiedades necesarias."""
    
    # Verificar conectividad
    if not nx.is_connected(graph):
        return False
    
    # Verificar que tenga al menos un starter, un depot y un fire
    node_types = [data.get('type') for _, data in graph.nodes(data=True)]
    
    required_types = ['starter', 'depot', 'fire']
    for req_type in required_types:
        if req_type not in node_types:
            return False
    
    # Verificar que todas las aristas tengan propiedades
    for u, v, data in graph.edges(data=True):
        if 'width' not in data or 'transit_time' not in data:
            return False
    
    return True

def save_graphs_to_files(graphs: List[nx.Graph], base_filename: str = "generated_graph", save_images: bool = True):
    """Guarda los grafos generados en archivos JSON y opcionalmente sus imágenes."""
    
    # Crear timestamp para organizar archivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear directorio para grafos JSON con timestamp
    config_dir = f"config_graph_{timestamp}"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Directorio creado: {config_dir}")
    
    # Directorio para imágenes si se solicita
    if save_images:
        image_dir = f"generated_graphs_images_{timestamp}"
    
    for i, graph in enumerate(graphs):
        filename = f"{base_filename}_{i+1}"
        json_filename = os.path.join(config_dir, f"{filename}.json")
        
        # Convertir el grafo a formato serializable
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Guardar información de nodos
        for node, data in graph.nodes(data=True):
            node_info = {'id': node, **data}
            graph_data['nodes'].append(node_info)
        
        # Guardar información de aristas
        for u, v, data in graph.edges(data=True):
            edge_info = {'source': u, 'target': v, **data}
            graph_data['edges'].append(edge_info)
        
        # Guardar en archivo JSON
        with open(json_filename, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Grafo {i+1} guardado como {json_filename}")
        
        # Guardar imagen si se solicita
        if save_images:
            visualize_and_save_graph(graph, filename, image_dir)

def load_graph_from_file(filename: str) -> nx.Graph:
    """Carga un grafo desde un archivo JSON."""
    
    with open(filename, 'r') as f:
        graph_data = json.load(f)
    
    graph = nx.Graph()
    
    # Cargar nodos
    for node_info in graph_data['nodes']:
        node_id = node_info.pop('id')
        graph.add_node(node_id, **node_info)
    
    # Cargar aristas
    for edge_info in graph_data['edges']:
        source = edge_info.pop('source')
        target = edge_info.pop('target')
        graph.add_edge(source, target, **edge_info)
    
    return graph

def visualize_and_save_graph(graph: nx.Graph, filename: str, save_dir: str = "generated_graphs_images"):
    """Visualiza y guarda una imagen del grafo."""
    
    # Crear directorio si no existe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directorio creado: {save_dir}")
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Generar layout del grafo
    pos = nx.spring_layout(graph, seed=42, k=1, iterations=50)
    
    # Definir colores para diferentes tipos de nodos
    node_colors = {
        'starter': 'yellow',
        'depot': 'blue', 
        'fire': 'red'
    }
    
    # Asignar colores a los nodos
    colors = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'starter')  # Default a starter si no tiene tipo
        colors.append(node_colors.get(node_type, 'yellow'))
    
    # Dibujar el grafo
    nx.draw(graph, pos, 
            node_color=colors, 
            node_size=800,
            with_labels=True, 
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            alpha=0.8)
    
    # Añadir etiquetas de aristas con información de ancho y tiempo
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        width = data.get('width', 'N/A')
        transit_time = data.get('transit_time', 'N/A')
        edge_labels[(u, v)] = f"w:{width}, t:{transit_time}"
    
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    # Añadir título con estadísticas del grafo
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    num_depots = len([n for n, d in graph.nodes(data=True) if d.get('type') == 'depot'])
    num_fires = len([n for n, d in graph.nodes(data=True) if d.get('type') == 'fire'])
    num_starters = len([n for n, d in graph.nodes(data=True) if d.get('type') == 'starter'])
    
    title = f"Grafo Generado\nNodos: {num_nodes} | Aristas: {num_edges}\n"
    title += f"Depósitos: {num_depots} | Incendios: {num_fires} | Starters: {num_starters}"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Añadir leyenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markersize=10, label='Starter'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Depósito'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='Incendio')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Guardar imagen
    filepath = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Cerrar figura para liberar memoria
    
    print(f"Imagen guardada: {filepath}")
    return filepath

# Función de ejemplo para generar y probar diferentes configuraciones
def generate_test_scenarios():
    """Genera varios escenarios de prueba con diferentes características."""
    
    scenarios = [
        {
            "name": "Escenario Básico",
            "connectivity": "medium",
            "num_depots": "medium", 
            "num_fires": "medium",
            "edge_widths": "medium",
            "transit_times": "medium"
        },
        {
            "name": "Escenario Desafiante - Alta Conectividad",
            "connectivity": "high",
            "num_depots": "low",
            "num_fires": "high", 
            "edge_widths": "low",
            "transit_times": "high"
        },
        {
            "name": "Escenario Rápido - Muchos Recursos",
            "connectivity": "high",
            "num_depots": "high",
            "num_fires": "low",
            "edge_widths": "high", 
            "transit_times": "low"
        },
        {
            "name": "Escenario Disperso - Baja Conectividad",
            "connectivity": "low",
            "num_depots": "medium",
            "num_fires": "medium",
            "edge_widths": "medium",
            "transit_times": "high"
        },
        {
            "name": "Escenario Intenso - Muchos Incendios",
            "connectivity": "medium",
            "num_depots": "low", 
            "num_fires": "high",
            "edge_widths": "low",
            "transit_times": "medium"
        }
    ]
    
    all_graphs = []
    
    for scenario in scenarios:
        print(f"\n=== Generando {scenario['name']} ===")
        graphs = generate_variable_graphs(
            num_graphs=1,  # Un grafo por escenario
            connectivity=scenario['connectivity'],
            num_depots=scenario['num_depots'], 
            num_fires=scenario['num_fires'],
            edge_widths=scenario['edge_widths'],
            transit_times=scenario['transit_times'],
            seed=42  # Para reproducibilidad
        )
        all_graphs.extend(graphs)
    
    return all_graphs

def plot_graph(graph: nx.Graph, ax: plt.Axes = None, title: str = "", with_labels: bool = True):
    """Dibuja el grafo en un eje dado de Matplotlib."""
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar el grafo
    pos = nx.spring_layout(graph, seed=42)  # Posiciones fijas para reproducibilidad
    nx.draw(graph, pos, ax=ax, with_labels=with_labels, node_color='lightblue', edge_color='gray', font_weight='bold', node_size=1000)
    
    # Dibujar etiquetas de ancho de arista
    edge_labels = nx.get_edge_attributes(graph, 'width')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
    
    # Configurar título y eliminar ejes
    ax.set_title(title)
    ax.axis('off')

def save_graph_plot(graph: nx.Graph, filename: str):
    """Guarda una imagen del grafo en un archivo."""
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Guardar figura
    plt.figure(figsize=(10, 8))
    plot_graph(graph, title="Grafo Generado")
    plt.savefig(filename, format='png')
    plt.close()
    print(f"Gráfico guardado como {filename}")

def load_and_plot_graph(filename: str):
    """Carga un grafo desde un archivo y lo dibuja."""
    
    graph = load_graph_from_file(filename)
    plt.figure(figsize=(10, 8))
    plot_graph(graph, title=f"Grafo desde {filename}")
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    print("=== Generador de Grafos para Extinción de Incendios ===\n")
    
    # Generar 5 grafos con configuración por defecto
    graphs = generate_variable_graphs(
        num_graphs=5,
        connectivity="medium",
        num_depots="medium",
        num_fires="medium", 
        edge_widths="medium",
        transit_times="medium",
        seed=42
    )
    
    # Guardar grafos en archivos
    save_graphs_to_files(graphs)
    
    print(f"\n✓ Se generaron {len(graphs)} grafos exitosamente")
    print("Los grafos han sido guardados en archivos JSON")
