import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from agents import QLearningAgent, SarsaAgent
import os

def train_agent(env, agent, n_episodes, is_sarsa=False):
    for _ in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        action = agent.choose_action(state)
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.choose_action(next_state)
            
            if is_sarsa:
                agent.update(state, action, reward, next_state, next_action)
            else:
                agent.update(state, action, reward, next_state)
                
            state = next_state
            action = next_action

def evaluate_agent(env, agent, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def generate_gif(env, agent, filename):
    state, _ = env.reset()
    done = False
    frames = []
    
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.choose_action(state, explore=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    # Append the final frame
    frame = env.render()
    frames.append(frame)
    
    # Save the gif
    imageio.mimsave(filename, frames, fps=4, loop=0)

def main():
    if not os.path.exists("img"):
        os.makedirs("img")

    env = gym.make("CliffWalking-v1")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Q-Learning
    ql_agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    train_agent(env, ql_agent, 2000, is_sarsa=False)
    
    # SARSA
    sarsa_agent = SarsaAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    train_agent(env, sarsa_agent, 2000, is_sarsa=True)
    
    # Evaluate 100 episodes
    ql_eval_rewards = evaluate_agent(env, ql_agent, 100)
    sarsa_eval_rewards = evaluate_agent(env, sarsa_agent, 100)
    
    # Plot evaluation rewards
    plt.figure(figsize=(10, 5))
    plt.plot(ql_eval_rewards, label="Q-Learning")
    plt.plot(sarsa_eval_rewards, label="SARSA")
    plt.title("Avaliação do CliffWalking (100 Episódios)")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.savefig("img/cliff_walking_eval.png")
    
    # Check constraints
    if min(ql_eval_rewards) <= -100:
        print("Warning: Q-Learning agent fell off the cliff!")
    if min(sarsa_eval_rewards) <= -100:
        print("Warning: SARSA agent fell off the cliff!")
    
    # Generate GIFs
    env_render = gym.make("CliffWalking-v1", render_mode="rgb_array")
    generate_gif(env_render, ql_agent, "img/cliff_walking_qlearning.gif")
    generate_gif(env_render, sarsa_agent, "img/cliff_walking_sarsa.gif")
    print("Avaliação do CliffWalking concluída. Gráficos e GIFs salvos.")

if __name__ == "__main__":
    main()
