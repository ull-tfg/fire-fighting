import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import time
import copy

from qmix import QMixer
from qmix_agent import QMIXAgent, EpisodeBuffer, Transition

class QMIXTrainer:
    def __init__(self, env, state_dim, global_state_dim, num_agents, vehicle_types, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.env = env
        self.num_agents = num_agents
        
        # Inicializar agentes
        self.agents = []
        for agent_id in range(num_agents):
            action_dim = env.action_space[agent_id].n
            vehicle_type = f"agent_{agent_id}"
            agent = QMIXAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                vehicle_type=vehicle_type,
                agent_id=agent_id
            )
            self.agents.append(agent)
        
        # Inicializar mixer de QMIX
        self.mixer = QMixer(
            num_agents=num_agents,
            state_dim=global_state_dim,
            device=self.device
        )
        
        # Experiencia compartida para entrenamiento centralizado
        self.replay_buffer = EpisodeBuffer(capacity=1000)
        self.batch_size = 8  # Episodios por batch
        
    def collect_episode(self, render=False):
        """Recolecta un episodio completo de experiencia"""
        states, info = self.env.reset()
        done = False
        episode_reward = 0
        agent_rewards = [0] * self.num_agents
        total_steps = 0
        
        # Cada agente mantiene su propia memoria de episodio
        for agent in self.agents:
            agent.episode_memory = []
            
        while not done and total_steps < self.env.max_steps:
            # Construir estado global (concatenación de todas las observaciones)
            global_state = np.concatenate(states)
            
            # Seleccionar acciones para todos los agentes
            actions = []
            for agent_id, agent in enumerate(self.agents):
                available_actions = [
                    self.env.all_actions.index(action) 
                    for action in self.env.agent_action_spaces[agent_id]['available']
                ]
                action = agent.select_action(states[agent_id], available_actions)
                actions.append(action)
            
            # Ejecutar acciones en el entorno
            next_states, rewards, terminated, truncated, info = self.env.step(actions)
            next_global_state = np.concatenate(next_states)
            done = terminated or truncated
            
            # Almacenar transiciones para cada agente
            for agent_id, agent in enumerate(self.agents):
                agent.store_transition(
                    states[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_states[agent_id],
                    done,
                    global_state,
                    next_global_state
                )
            
            # Actualizar contadores y estado
            episode_reward += sum(rewards)
            for agent_id in range(self.num_agents):
                agent_rewards[agent_id] += rewards[agent_id]
            states = next_states
            total_steps += 1
            
            if render:
                self.env.render()
                
        # Almacenar episodio completo en el buffer
        episode_data = [
            (
                [agent.episode_memory[i].state for agent in self.agents],
                [agent.episode_memory[i].action for agent in self.agents],
                [agent.episode_memory[i].reward for agent in self.agents],
                [agent.episode_memory[i].next_state for agent in self.agents],
                agent.episode_memory[i].done,
                agent.episode_memory[i].global_state,
                agent.episode_memory[i].next_global_state
            )
            for i in range(len(self.agents[0].episode_memory))
        ]
        
        if len(episode_data) > 0:
            self.replay_buffer.push(episode_data)
        
        return episode_reward, agent_rewards, info['fires_extinguished'], total_steps
    
    def train(self):
        """Entrena a los agentes con QMIX usando episodios almacenados"""
        if len(self.replay_buffer) < 4:  # Necesitamos suficientes episodios
            return 0
        
        # Muestrear batch de episodios
        batch_episodes = self.replay_buffer.sample(self.batch_size)
        
        loss_sum = 0
        batch_count = 0
        
        # Procesar cada episodio
        for episode in batch_episodes:
            # Procesar cada paso de tiempo en el episodio
            for t in range(len(episode)):
                batch_count += 1
                states_t, actions_t, rewards_t, next_states_t, done_t, global_state_t, next_global_state_t = episode[t]
                
                # Obtener valores Q de cada agente para el estado actual
                q_values = []
                for agent_id, agent in enumerate(self.agents):
                    state_tensor = torch.FloatTensor(np.array([states_t[agent_id]])).to(self.device)
                    all_q_values = agent.policy_net(state_tensor)
                    q_value = all_q_values[0, actions_t[agent_id]].unsqueeze(0)
                    q_values.append(q_value)
                q_values = torch.stack(q_values, dim=1)  # [batch=1, num_agents]
                
                # Valores Q objetivo del siguiente estado
                if not done_t:
                    # Obtener valores Q máximos del siguiente estado
                    target_q_values = []
                    for agent_id, agent in enumerate(self.agents):
                        next_state_tensor = torch.FloatTensor(np.array([next_states_t[agent_id]])).to(self.device)
                        target_q_vals = agent.target_net(next_state_tensor)
                        target_q_values.append(target_q_vals.max(1)[0].detach())
                    target_q_values = torch.stack(target_q_values, dim=1)  # [batch=1, num_agents]
                    
                    # Mezclar valores Q objetivo usando el mixer objetivo
                    global_state_tensor = torch.FloatTensor(np.array([next_global_state_t])).to(self.device)
                    target_q_tot = self.mixer.target_mix(target_q_values, global_state_tensor)
                    
                    # Calcular el objetivo en base a la recompensa y el valor futuro
                    rewards_tensor = torch.FloatTensor(np.array([rewards_t])).to(self.device).sum(dim=1, keepdim=True)
                    y = rewards_tensor + self.agents[0].gamma * target_q_tot
                else:
                    # Si es el estado final, solo usar la recompensa
                    rewards_tensor = torch.FloatTensor(np.array([rewards_t])).to(self.device).sum(dim=1, keepdim=True)
                    y = rewards_tensor
                
                # Mezclar valores Q actuales
                global_state_tensor = torch.FloatTensor(np.array([global_state_t])).to(self.device)
                q_tot = self.mixer.mix(q_values, global_state_tensor)
                
                # Calcular pérdida
                loss = F.mse_loss(q_tot, y.detach())
                
                # Optimizar
                self.mixer.optimizer.zero_grad()
                for agent in self.agents:
                    agent.optimizer.zero_grad()
                
                loss.backward()
                
                # Aplicar gradientes
                for agent in self.agents:
                    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 10)
                    agent.optimizer.step()
                
                self.mixer.optimizer.step()
                
                loss_sum += loss.item()
                
        # Actualizar redes objetivo
        for agent in self.agents:
            agent.update_target_net()
        self.mixer.update_target_mixer()
        
        return loss_sum / max(1, batch_count)
    
    def train_qmix(self, num_episodes=2000, max_steps=100, eval_freq=10, render_training=False):
        """Método principal de entrenamiento usando QMIX"""
        start_time = time.time()
        
        # Métricas de seguimiento
        episode_rewards = []
        total_fires_extinguished = []
        steps_per_episode = []
        agent_rewards = {i: [] for i in range(self.num_agents)}
        eval_rewards = []
        eval_fires_extinguished = []
        losses = []
        
        for episode in range(num_episodes):
            # Recolectar episodio
            render_this_episode = (episode % 100 == 0) and render_training
            episode_reward, agent_episode_rewards, fires_extinguished, steps = self.collect_episode(render=render_this_episode)
            
            # Entrenar con experiencias recolectadas
            loss = self.train()
            
            # Almacenar métricas
            episode_rewards.append(episode_reward)
            total_fires_extinguished.append(fires_extinguished)
            steps_per_episode.append(steps)
            losses.append(loss)
            
            # Almacenar recompensas individuales de agentes
            for agent_id in range(self.num_agents):
                agent_rewards[agent_id].append(agent_episode_rewards[agent_id])
            
            # Mostrar progreso
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_fires = np.mean(total_fires_extinguished[-10:])
                avg_loss = np.mean([l for l in losses[-10:] if l != 0])
                avg_agent_rewards = {i: np.mean(agent_rewards[i][-10:]) for i in range(self.num_agents)}
                
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                elapsed = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                print(f"Episode {episode+1}/{num_episodes}, Total Reward: {avg_reward:.2f}, "
                      f"Loss: {avg_loss:.5f}, Fires: {avg_fires:.1f}/{len(self.env.fire_nodes)}, "
                      f"Elapsed: {elapsed}")
                
                # Imprimir métricas individuales de agentes
                for agent_id in range(self.num_agents):
                    print(f"  Agent {agent_id} Avg Reward: {avg_agent_rewards[agent_id]:.2f}")
            
            # Evaluación
            if (episode + 1) % eval_freq == 0:
                eval_reward, eval_fires = self.evaluate(num_eval_episodes=5)
                eval_rewards.append(eval_reward)
                eval_fires_extinguished.append(eval_fires)
                
                print(f"Evaluation - Avg Reward: {eval_reward:.2f}, "
                      f"Avg Fires Extinguished: {eval_fires:.1f}/{len(self.env.fire_nodes)}")
        
        # Resultados finales
        print("\nIndividual Agent Performance:")
        for agent_id in range(self.num_agents):
            avg_reward = np.mean(agent_rewards[agent_id][-50:])
            print(f"  Agent {agent_id} - Avg Reward in Last 50 Episodes: {avg_reward:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'fires_extinguished': total_fires_extinguished,
            'steps_per_episode': steps_per_episode,
            'agent_rewards': agent_rewards,
            'eval_rewards': eval_rewards,
            'eval_fires_extinguished': eval_fires_extinguished,
            'losses': losses
        }
    
    def evaluate(self, num_eval_episodes=5):
        """Evalúa el rendimiento de los agentes sin exploración"""
        eval_rewards = []
        eval_fires = []
        
        # Guardar y restaurar tasas de exploración
        original_eps = [agent.epsilon for agent in self.agents]
        for agent in self.agents:
            agent.epsilon = 0.00  # Exploración mínima durante evaluación
        
        for _ in range(num_eval_episodes):
            states, info = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < self.env.max_steps:
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    available_actions = [
                        self.env.all_actions.index(action) 
                        for action in self.env.agent_action_spaces[agent_id]['available']
                    ]
                    action = agent.select_action(states[agent_id], available_actions)
                    actions.append(action)
                
                next_states, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                total_reward += sum(rewards)
                states = next_states
                steps += 1
            
            eval_rewards.append(total_reward)
            eval_fires.append(info['fires_extinguished'])
        
        # Restaurar tasas de exploración originales
        for agent, eps in zip(self.agents, original_eps):
            agent.epsilon = eps
        
        return np.mean(eval_rewards), np.mean(eval_fires)