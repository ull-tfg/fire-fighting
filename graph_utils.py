import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_graph(fires=8, tanks=3, starters=3):
    """
    Genera un grafo completamente conectado con nodos representando incendios y tanques.
    
    Parámetros:
    - fires (int): Número total de nodos de incendios
    - tanks (int): Número de nodos iniciales (tanques)
    - starters (int): Número de nodos iniciales
    
    Retorna:
    - nx.Graph: Grafo completamente conectado
    """
    # Crear un grafo vacío
    G = nx.Graph()
    
    # Agregar nodos de tanques como nodos iniciales
    for i in range(tanks):
        G.add_node(f'tank_{i}', type='tank')
    
    # Agregar nodos de incendios
    for i in range(fires):
        G.add_node(f'fire_{i}', type='fire', water_to_extinguish=random.randint(50, 250))

    # Agregar nodos iniciales
    for i in range(starters):
        G.add_node(f'starter_{i}', type='starter')
    
    # Conectar todos los nodos entre sí
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                # Generar un tiempo aleatorio entre 1 y 20
                transit_time = random.randint(1, 3)
                G.add_edge(node1, node2, transit_time=transit_time)
    
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
        'tank': 'blue',
        'fire': 'red',
        'starter': 'yellow'
    }
    
    # Asignar colores a los nodos
    colors = [node_colors[G.nodes[node]['type']] for node in G.nodes()]
    
    # Dibujar el grafo
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500)

    # Dibujar etiquetas de las aristas
    edge_labels = nx.get_edge_attributes(G, 'transit_time')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Retornar la figura
    return fig

def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def print_results(start_time, episode_rewards, fires_extinguished_per_episode, steps_per_episode):
    # Resumen final y visualización
    print("\nEntrenamiento completado!")
    print(f"Tiempo total: {format_time(time.time() - start_time)}")
    print(f"Reward promedio final: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Fuegos apagados promedio final: {np.mean(fires_extinguished_per_episode[-10:]):.1f}")
    
    # Graficar resultados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Rewards por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Reward Total')
    
    plt.subplot(1, 3, 2)
    plt.plot(fires_extinguished_per_episode)
    plt.title('Fuegos Extinguidos por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Fuegos Extinguidos')
    
    plt.subplot(1, 3, 3)
    plt.plot(steps_per_episode)
    plt.title('Pasos por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Número de Pasos')
    
    plt.tight_layout()
    plt.show()