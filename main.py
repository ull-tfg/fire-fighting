import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque
import networkx as nx

# Import custom modules
from environment import FirefightingEnv
from dqn import DQN
from agent import DQNAgent
from graph_utils import generate_graph, visualize_graph, format_time, print_results

def train_multi_agent(env, agents, num_episodes=500, max_steps=200, eval_freq=10, render_training=False):
    """
    Train multiple DQN agents in the firefighting environment.
    
    Args:
        env: The multi-agent environment
        agents: List of DQN agents (one per firefighter)
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        eval_freq: Frequency of evaluation episodes
        render_training: Whether to render during training
    
    Returns:
        Dictionary of training metrics for all agents
    """
    start_time = time.time()
    num_agents = len(agents)
    
    # Metrics tracking
    episode_rewards = []
    total_fires_extinguished = []
    steps_per_episode = []
    agent_rewards = {i: [] for i in range(num_agents)}
    eval_rewards = []
    eval_fires_extinguished = []
    
    for episode in range(num_episodes):
        states, info = env.reset()
        episode_reward = 0
        agent_episode_rewards = [0] * num_agents
        render_training = False
        if episode % 100 == 0:
            render_training = True
        
        for step in range(max_steps):
            # Select actions for all agents
            actions = []
            for agent_id, agent in enumerate(agents):
                state_tensor = torch.FloatTensor(states[agent_id]).unsqueeze(0)
                # Obtener índices de acciones disponibles para este agente
                available_actions = [
                    env.all_actions.index(action) 
                    for action in env.agent_action_spaces[agent_id]['available']
                ]
                action = agent.select_action(state_tensor, available_actions)
                actions.append(action.item())
            # Take actions in environment
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store transitions and learn for each agent
            for agent_id, agent in enumerate(agents):
                # Store transition in memory
                agent.memory.push(states[agent_id], actions[agent_id], 
                                  next_states[agent_id], rewards[agent_id], done)
                # Learn from experience
                agent.learn(agent.batch_size)
                # Soft update target network
                if random.random() < agent.target_update_freq:
                    agent.update_target_model()
                # Update agent rewards
                agent_episode_rewards[agent_id] += rewards[agent_id]
            
            # Update total episode reward
            episode_reward += sum(rewards)
            # Update states for next iteration
            states = next_states
            if render_training:
                env.render()
            if done:
                break
    
        # Store metrics
        episode_rewards.append(episode_reward)
        total_fires_extinguished.append(info['fires_extinguished'])
        steps_per_episode.append(step + 1)
        
        # Store individual agent rewards
        for agent_id in range(num_agents):
            agent_rewards[agent_id].append(agent_episode_rewards[agent_id])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_fires = np.mean(total_fires_extinguished[-10:])
            avg_agent_rewards = {i: np.mean(agent_rewards[i][-10:]) for i in range(num_agents)}
            elapsed = format_time(time.time() - start_time)
            
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {avg_reward:.2f}, "
                  f"Fires: {avg_fires:.1f}/{len(env.fire_nodes)}, Elapsed: {elapsed}")
            
            # Print individual agent metrics
            for agent_id in range(num_agents):
                print(f"  Agent {agent_id} Avg Reward: {avg_agent_rewards[agent_id]:.2f}")
        
        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_fires = evaluate_agents(env, agents, num_eval_episodes=5)
            eval_rewards.append(eval_reward)
            eval_fires_extinguished.append(eval_fires)
            
            print(f"Evaluation - Avg Reward: {eval_reward:.2f}, "
                  f"Avg Fires Extinguished: {eval_fires:.1f}/{len(env.fire_nodes)}")

    # Print final results
    print_results(start_time, episode_rewards, total_fires_extinguished, steps_per_episode)
    print("Individual Agent Performance:")
    for agent_id in range(num_agents):
        avg_reward = np.mean(agent_rewards[agent_id][-50:])
        print(f"  Agent {agent_id} - Avg Reward in Last 50 Episodes: {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'fires_extinguished': total_fires_extinguished,
        'steps_per_episode': steps_per_episode,
        'agent_rewards': agent_rewards,
        'eval_rewards': eval_rewards,
        'eval_fires_extinguished': eval_fires_extinguished
    }

def evaluate_agents(env, agents, num_eval_episodes=5):
    """
    Evaluate the performance of the agents without exploration.
    
    Args:
        env: The environment
        agents: List of DQN agents
        num_eval_episodes: Number of evaluation episodes
    
    Returns:
        Tuple of (average_reward, average_fires_extinguished)
    """
    eval_rewards = []
    eval_fires = []
    
    # Save and restore exploration rates
    original_eps = [agent.epsilon for agent in agents]
    for agent in agents:
        agent.epsilon = 0.00  # Minimal exploration during evaluation
    
    for _ in range(num_eval_episodes):
        states, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            actions = []
            for agent_id, agent in enumerate(agents):
                state_tensor = torch.FloatTensor(states[agent_id]).unsqueeze(0)
                available_actions = [
                    env.all_actions.index(action) 
                    for action in env.agent_action_spaces[agent_id]['available']
                ]
                action = agent.select_action(state_tensor, available_actions)
                actions.append(action.item())
            
            next_states, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            total_reward += sum(rewards)
            states = next_states
        
        eval_rewards.append(total_reward)
        eval_fires.append(info['fires_extinguished'])
    
    # Restore original exploration rates
    for agent, eps in zip(agents, original_eps):
        agent.epsilon = eps
    
    return np.mean(eval_rewards), np.mean(eval_fires)

def visualize_multi_agent_performance(metrics, num_agents):
    """
    Visualize the multi-agent team's performance over time.
    
    Args:
        metrics: Dictionary of training metrics
        num_agents: Number of agents in the team
    """
    plt.figure(figsize=(16, 12))
    
    # Plot team rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Team Total Rewards per Episode')
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
    
    # Plot evaluation metrics
    plt.subplot(3, 2, 6)
    eval_x = list(range(0, len(metrics['episode_rewards']), 10))
    eval_x = eval_x[:len(metrics['eval_rewards'])]
    
    plt.plot(eval_x, metrics['eval_rewards'], label='Eval Rewards', color='blue')
    plt.plot(eval_x, metrics['eval_fires_extinguished'], 
             label='Eval Fires Extinguished', color='red')
    plt.title('Evaluation Metrics')
    plt.xlabel('Episode')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi_agent_training_results.png')
    plt.show()

def plot_agent_cooperation(metrics, num_agents):
    """
    Plot the cooperation metrics between agents.
    
    Args:
        metrics: Dictionary of training metrics
        num_agents: Number of agents in the team
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate the correlation between agent rewards over time
    corr_matrix = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                rewards_i = metrics['agent_rewards'][i]
                rewards_j = metrics['agent_rewards'][j]
                corr = np.corrcoef(rewards_i, rewards_j)[0, 1]
                corr_matrix[i, j] = corr
    
    # Plot correlation matrix
    plt.subplot(1, 2, 1)
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Agent Reward Correlation')
    plt.xlabel('Agent ID')
    plt.ylabel('Agent ID')
    plt.xticks(range(num_agents))
    plt.yticks(range(num_agents))
    
    # Plot the contribution to team success
    plt.subplot(1, 2, 2)
    
    # Calculate the average contribution of each agent
    episode_count = len(metrics['episode_rewards'])
    agent_contribution = np.zeros(num_agents)
    
    for agent_id in range(num_agents):
        total_agent_reward = sum(metrics['agent_rewards'][agent_id])
        total_team_reward = sum(metrics['episode_rewards'])
        if total_team_reward > 0:
            agent_contribution[agent_id] = total_agent_reward / total_team_reward * 100
        else:
            agent_contribution[agent_id] = 0
    
    # Create bar plot
    plt.bar(range(num_agents), agent_contribution)
    plt.title('Agent Contribution to Team Reward')
    plt.xlabel('Agent ID')
    plt.ylabel('Contribution (%)')
    plt.xticks(range(num_agents))
    
    plt.tight_layout()
    plt.savefig('agent_cooperation.png')
    plt.show()

if __name__ == "__main__":
    steps = 300  # Maximum steps per episode
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
        1: {'capacity': 100, 'width': 1},  # Large capacity, narrow spray
        2: {'capacity': 500, 'width': 4}   # Small capacity, wide spray
    }
    # Update environment with vehicle types
    env = FirefightingEnv(graph=graph, max_steps=steps, num_agents=num_agents, 
                        vehicle_types=vehicle_types)
    
    # Create a list of agents
    agents = []
    for agent_id in range(num_agents):
        state_dim = env.observation_space[agent_id].shape[0]
        action_dim = env.action_space[agent_id].n
        vehicle_type = f"agent_{agent_id}"

        agent = DQNAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            vehicle_type=vehicle_type,
        )
        agents.append(agent)
    
    # Train the multi-agent team
    metrics = train_multi_agent(
        env=env, 
        agents=agents, 
        num_episodes=1000, 
        max_steps=steps,
        eval_freq=10, 
        render_training=False
    )
    
    # Visualize results
    visualize_multi_agent_performance(metrics, num_agents)
    plot_agent_cooperation(metrics, num_agents)

    # View render environment with trained agents
    train_multi_agent(
        env=env, 
        agents=agents, 
        num_episodes=1, 
        max_steps=steps,
        eval_freq=1,
        render_training=True
    )

    # Importa la nueva clase
    from interactive_viz import visualize_interactive
    print("\nIniciando visualización interactiva...")
    visualize_interactive(env, agents)