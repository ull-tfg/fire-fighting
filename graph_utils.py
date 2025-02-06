import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_graph(num_incendios=10, num_estanques=4):
    G = nx.Graph()
    
    # Crear nodos
    incendios = [f"incendio_{i}" for i in range(num_incendios)]
    estanques = [f"estanque_{i}" for i in range(num_estanques)]
    nodos = incendios + estanques
    
    # Añadir nodos con atributos
    for nodo in nodos:
        if "incendio" in nodo:
            G.add_node(nodo, tipo='incendio', water_to_extinguish=random.randint(50, 100))
        else:
            G.add_node(nodo, tipo='estanque', water_capacity=random.randint(1000, 2000))
    
    # Crear una estructura base conectada
    # Primero, conectar todos los nodos en una cadena para garantizar conectividad
    for i in range(len(nodos) - 1):
        G.add_edge(nodos[i], nodos[i + 1], 
                  ancho=random.choice([2, 3]),  # Evitar caminos muy estrechos
                  tiempo_viaje=random.randint(1, 5))
    
    # Asegurar que cada incendio tiene al menos un camino a un estanque
    for incendio in incendios:
        if not any(nx.has_path(G, incendio, estanque) for estanque in estanques):
            estanque = random.choice(estanques)
            G.add_edge(incendio, estanque, 
                      ancho=random.choice([2, 3]),
                      tiempo_viaje=random.randint(1, 5))
    
    # Añadir conexiones adicionales para mayor conectividad
    for _ in range(len(nodos)):
        nodo1 = random.choice(nodos)
        nodo2 = random.choice(nodos)
        if nodo1 != nodo2 and not G.has_edge(nodo1, nodo2):
            G.add_edge(nodo1, nodo2, 
                      ancho=random.choice([2, 3]),
                      tiempo_viaje=random.randint(1, 5))
    
    # Validar el grafo
    validate_graph(G, incendios, estanques)
    
    return G

def validate_graph(G, incendios, estanques):
    # Verificar conectividad
    if not nx.is_connected(G):
        raise ValueError("El grafo no está completamente conectado")
    
    # Verificar acceso a estanques
    for incendio in incendios:
        if not any(nx.has_path(G, incendio, estanque) for estanque in estanques):
            raise ValueError(f"El incendio {incendio} no tiene acceso a ningún estanque")
    
    # Verificar anchos de camino
    for u, v, data in G.edges(data=True):
        if 'ancho' not in data or data['ancho'] < 1:
            raise ValueError(f"Camino entre {u} y {v} no tiene ancho válido")
        if 'tiempo_viaje' not in data or data['tiempo_viaje'] < 1:
            raise ValueError(f"Camino entre {u} y {v} no tiene tiempo de viaje válido")

def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Dibujar nodos
    incendios = [n for n, attr in G.nodes(data=True) if attr['tipo'] == 'incendio']
    estanques = [n for n, attr in G.nodes(data=True) if attr['tipo'] == 'estanque']
    
    nx.draw_networkx_nodes(G, pos, nodelist=incendios, node_color='red', 
                          node_size=500, label='Incendios')
    nx.draw_networkx_nodes(G, pos, nodelist=estanques, node_color='blue', 
                          node_size=500, label='Estanques')
    
    # Dibujar aristas con información
    edge_labels = {(u, v): f"A:{d['ancho']}\nT:{d['tiempo_viaje']}" 
                  for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Etiquetas de nodos con información
    labels = {}
    for node in G.nodes():
        if 'incendio' in node:
            water = G.nodes[node]['water_to_extinguish']
            labels[node] = f"{node}\n({water})"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Grafo del Entorno de Bomberos")
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    
    return plt

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