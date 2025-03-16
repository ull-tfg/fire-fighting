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

def train_agent(env, agent, num_episodes=500, max_steps=200, eval_freq=10, render_training=False):
    """
    Train the DQN agent in the firefighting environment.
    
    Args:
        env: The environment to train in
        agent: The DQN agent
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        eval_freq: Frequency of evaluation episodes
        render_training: Whether to render during training
    
    Returns:
        Dictionary of training metrics
    """
    start_time = time.time()
    
    # Metrics tracking
    episode_rewards = []
    fires_extinguished_per_episode = []
    steps_per_episode = []
    eval_rewards = []
    eval_fires_extinguished = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        state_history = []
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Select action
            action = agent.select_action(state_tensor)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # Store transition in memory
            agent.memory.push(state, action.item(), next_state, reward, done)
            
            # Learn from experience
            agent.learn(agent.batch_size)
            
            # Soft update target network
            if random.random() < agent.target_update_freq:
                agent.update_target_model()
            
            # store state for next iteration
            state_history.append(state)
            
            state = next_state
            episode_reward += reward
            
            if render_training:
                env.render()
            
            if done:
                if truncated and episode > 300:
                    print("Episode truncated at max steps")
                    for state in state_history:
                        print(state)
                        print('\n-------------------\n')
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        fires_extinguished_per_episode.append(info['fires_extinguished'])
        steps_per_episode.append(step + 1)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_fires = np.mean(fires_extinguished_per_episode[-10:])
            elapsed = format_time(time.time() - start_time)
            
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                 f"Avg Fires: {avg_fires:.1f}/{len(env.fire_nodes)}, "
                 f"Elapsed: {elapsed}")

    # Print final results
    print_results(start_time, episode_rewards, fires_extinguished_per_episode, steps_per_episode)
    
    return {
        'episode_rewards': episode_rewards,
        'fires_extinguished': fires_extinguished_per_episode,
        'steps_per_episode': steps_per_episode,
        'eval_rewards': eval_rewards,
        'eval_fires_extinguished': eval_fires_extinguished
    }


def visualize_agent_performance(metrics):
    """
    Visualize the agent's performance over time.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot smoothed rewards
    plt.subplot(2, 2, 2)
    window_size = 10
    smoothed_rewards = np.convolve(metrics['episode_rewards'], 
                                  np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Rewards (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # Plot fires extinguished
    plt.subplot(2, 2, 3)
    plt.plot(metrics['fires_extinguished'])
    plt.title('Fires Extinguished per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Fires Extinguished')
    
    # Plot evaluation metrics
    plt.subplot(2, 2, 4)
    eval_x = list(range(0, len(metrics['episode_rewards']), 10))[:len(metrics['eval_rewards'])]
    
    plt.plot(eval_x, metrics['eval_rewards'], label='Eval Rewards')
    plt.title('Evaluation Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    steps = 300
    
    # Create environment with a random graph
    graph = generate_graph(num_incendios=8, num_estanques=3)
    
    # Visualize the graph
    plt_graph = visualize_graph(graph)
    plt_graph.savefig('environment_graph.png')
    plt_graph.show()
    
    # Create environment
    env = FirefightingEnv(graph=graph, max_steps=steps)
    
    # Set up the agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    vehicle_type = "standard"  # placeholder
    agent = DQNAgent(state_dim, action_dim, vehicle_type)
    
    # Train the agent
    metrics = train_agent(env, agent, num_episodes=500, max_steps=steps, 
                        eval_freq=10, render_training=False)
    
    # Visualize results
    visualize_agent_performance(metrics)