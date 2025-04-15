import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def create_default_exact_graph(filepath="exact_graph_config.json"):
    """
    Crea un archivo de configuración con un grafo exacto predefinido.
    Este grafo se utilizará exactamente como está definido, sin elementos aleatorios.
    """
    exact_config = {
        "nodes": {
            "fires": {
                "fire_0": {"water_to_extinguish": 50},
                "fire_1": {"water_to_extinguish": 30},
                "fire_2": {"water_to_extinguish": 40},
                "fire_3": {"water_to_extinguish": 60},
                "fire_4": {"water_to_extinguish": 45}
            },
            "depots": {
                "depot_0": {},
                "depot_1": {},
                "depot_2": {}
            },
            "starters": {
                "starter_0": {},
                "starter_1": {}
            }
        },
        "edges": [
            {"source": "starter_0", "target": "fire_0", "transit_time": 2, "width": 3},
            {"source": "starter_0", "target": "depot_0", "transit_time": 1, "width": 5},
            {"source": "starter_1", "target": "fire_1", "transit_time": 2, "width": 4},
            {"source": "starter_1", "target": "depot_1", "transit_time": 1, "width": 5},
            {"source": "fire_0", "target": "fire_1", "transit_time": 3, "width": 2},
            {"source": "fire_1", "target": "fire_2", "transit_time": 2, "width": 3},
            {"source": "fire_2", "target": "fire_3", "transit_time": 2, "width": 2},
            {"source": "fire_3", "target": "fire_4", "transit_time": 2, "width": 3},
            {"source": "fire_4", "target": "depot_2", "transit_time": 3, "width": 4},
            {"source": "depot_0", "target": "depot_1", "transit_time": 3, "width": 5},
            {"source": "depot_1", "target": "depot_2", "transit_time": 4, "width": 5},
            {"source": "starter_0", "target": "fire_2", "transit_time": 4, "width": 2},
            {"source": "depot_0", "target": "fire_2", "transit_time": 2, "width": 3},
            {"source": "depot_1", "target": "fire_3", "transit_time": 3, "width": 3},
            {"source": "fire_0", "target": "fire_3", "transit_time": 5, "width": 1}
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(exact_config, f, indent=4)
    
    print(f"Archivo de configuración exacta creado en {filepath}")
    return exact_config

def load_exact_graph(filepath="exact_graph_config.json"):
    """
    Carga un grafo exacto desde un archivo de configuración.
    """
    if not os.path.exists(filepath):
        print(f"Archivo de configuración no encontrado. Creando configuración por defecto en {filepath}")
        return create_default_exact_graph(filepath)
    
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    return config

def generate_exact_graph(config=None, config_file="exact_graph_config.json"):
    """
    Genera un grafo exacto según la configuración proporcionada.
    """
    if config is None:
        config = load_exact_graph(config_file)
    
    # Crear grafo vacío
    G = nx.Graph()
    
    # Añadir nodos de fuego con sus propiedades específicas
    for fire_id, properties in config["nodes"]["fires"].items():
        G.add_node(fire_id, type='fire', **properties)
    
    # Añadir nodos de tanque
    for depot_id, properties in config["nodes"]["depots"].items():
        G.add_node(depot_id, type='depot', **properties)
    
    # Añadir nodos de inicio
    for starter_id, properties in config["nodes"]["starters"].items():
        G.add_node(starter_id, type='starter', **properties)
    
    # Añadir aristas con sus propiedades específicas
    for edge in config["edges"]:
        G.add_edge(
            edge["source"], 
            edge["target"], 
            transit_time=edge["transit_time"], 
            width=edge["width"]
        )
    
    return G

def visualize_graph(G):
    """
    Visualiza un grafo utilizando la librería NetworkX.
    
    Parámetros:
    - G (nx.Graph): Grafo a visualizar
    
    Retorna:
    - matplotlib.figure.Figure: Figura del grafo
    """
    # Crear una nueva figura con un tamaño específico
    fig = plt.figure(figsize=(12, 8))
    
    # Crear un layout para el grafo
    pos = nx.spring_layout(G, seed=42)  # seed para reproducibilidad
    
    # Crear un diccionario de colores para los nodos
    node_colors = {
        'depot': 'blue',
        'fire': 'red',
        'starter': 'yellow'
    }
    
    # Asignar colores a los nodos
    colors = [node_colors[G.nodes[node]['type']] for node in G.nodes()]
    
    # Dibujar el grafo
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500)

    # Dibujar etiquetas de las aristas
    edge_labels = {}
    for u, v, attrs in G.edges(data=True):
        edge_labels[(u, v)] = f"t:{attrs['transit_time']}, w:{attrs['width']}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Retornar la figura
    return fig

def visualize_exact_graph(config=None, config_file="exact_graph_config.json", show=True):
    """
    Genera y visualiza el grafo exacto basado en la configuración.
    """
    G = generate_exact_graph(config, config_file)
    fig = visualize_graph(G)
    if show:
        plt.show()
    return G, fig

def save_current_graph_as_exact(G, filepath="exact_graph_config.json"):
    """
    Toma un grafo existente y lo guarda como un archivo de configuración exacta.
    Útil para guardar un grafo generado aleatoriamente que dio buenos resultados.
    """
    config = {
        "nodes": {
            "fires": {},
            "depots": {},
            "starters": {}
        },
        "edges": []
    }
    
    # Guardar nodos con sus propiedades
    for node, attrs in G.nodes(data=True):
        node_type = attrs['type']
        
        if node_type == 'fire':
            config["nodes"]["fires"][node] = {"water_to_extinguish": attrs.get("water_to_extinguish", 50)}
        elif node_type == 'depot':
            config["nodes"]["depots"][node] = {}
        elif node_type == 'starter':
            config["nodes"]["starters"][node] = {}
    
    # Guardar aristas con sus propiedades
    for source, target, attrs in G.edges(data=True):
        edge = {
            "source": source,
            "target": target,
            "transit_time": attrs.get("transit_time", 1),
            "width": attrs.get("width", 1)
        }
        config["edges"].append(edge)
    
    # Guardar en archivo
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Grafo actual guardado como configuración exacta en {filepath}")
    return config