import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx

# Import custom modules
from environment import FirefightingEnv
from graph_utils import generate_graph, visualize_graph, format_time
from qmix_trainer import QMIXTrainer

def visualize_qmix_performance(metrics, num_agents):
    """Visualizar el rendimiento del equipo multi-agente QMIX a lo largo del tiempo."""
    plt.figure(figsize=(16, 12))
    
    # Plot team rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Team Total Rewards per Episode (QMIX)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot smoothed team rewards
    plt.subplot(3, 2, 2)
    window_size = 10
    smoothed_rewards = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Team Rewards (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # Plot fires extinguished
    plt.subplot(3, 2, 3)
    plt.plot(metrics['fires_extinguished'])
    plt.title('Fires Extinguished per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Fires Extinguished')
    
    # Plot steps per episode
    plt.subplot(3, 2, 4)
    plt.plot(metrics['steps_per_episode'])
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot individual agent rewards
    plt.subplot(3, 2, 5)
    for agent_id in range(num_agents):
        plt.plot(metrics['agent_rewards'][agent_id], 
                 label=f'Agent {agent_id}', alpha=0.7)
    plt.title('Individual Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot losses
    plt.subplot(3, 2, 6)
    filtered_losses = [l if l != 0 else np.nan for l in metrics['losses']]
    plt.plot(filtered_losses, color='red')
    plt.title('QMIX Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('qmix_training_results.png')
    plt.show()

if __name__ == "__main__":
    steps = 100  # Maximum steps per episode
    num_agents = 3  # Number of firefighting agents
    
    # Create environment with a random graph
    graph = generate_graph()
    # Visualize the graph
    plt_graph = visualize_graph(graph)
    plt_graph.savefig('environment_graph.png')
    plt.show()

    # Set up the agents with different vehicle types for heterogeneity
    vehicle_types = {
        0: {'capacity': 200, 'width': 2},  # Standard vehicle
        1: {'capacity': 100, 'width': 1},  # Small capacity, narrow spray
        2: {'capacity': 500, 'width': 4}   # Large capacity, wide spray
    }
    
    # Update environment with vehicle types
    env = FirefightingEnv(graph=graph, max_steps=steps, num_agents=num_agents, 
                        vehicle_types=vehicle_types)
    
    # Determine state dimensions (local and global)
    state_dim = env.observation_space[0].shape[0]
    global_state_dim = state_dim * num_agents
    
    # Initialize the QMIX trainer
    trainer = QMIXTrainer(
        env=env,
        state_dim=state_dim,
        global_state_dim=global_state_dim,
        num_agents=num_agents,
        vehicle_types=vehicle_types
    )
    
    # Train using QMIX
    metrics = trainer.train_qmix(
        num_episodes=300,
        max_steps=steps,
        eval_freq=10,
        render_training=False
    )
    
    # Visualize results
    visualize_qmix_performance(metrics, num_agents)
    
    # View render environment with trained agents
    print("\nDemostrando agentes entrenados...")
    trainer.collect_episode(render=True)
    
    # Import interactive visualization if available
    try:
        from interactive_viz import visualize_interactive
        print("\nIniciando visualización interactiva...")
        visualize_interactive(env, trainer.agents)
    except ImportError:
        print("\nLa visualización interactiva no está disponible.")