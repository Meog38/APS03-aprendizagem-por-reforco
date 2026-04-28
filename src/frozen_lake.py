import gymnasium as gym
import numpy as np
from tqdm import tqdm
from agents import QLearningAgent, SarsaAgent
import os

def train_agent(env, agent, n_episodes, is_sarsa=False):
    # Use an epsilon decay to ensure it learns the optimal policy in the slippery environment
    initial_epsilon = agent.epsilon
    min_epsilon = 0.01
    decay_rate = 0.001
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        action = agent.choose_action(state)
        
        agent.epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
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
    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1.0:
                successes += 1
    return successes

def print_q_table_policy(q_table, env_desc):
    actions = ['L', 'D', 'R', 'U']
    policy = []
    for s in range(q_table.shape[0]):
        row = s // env_desc.shape[1]
        col = s % env_desc.shape[1]
        if env_desc[row, col] in [b'H', b'G']:
            policy.append(env_desc[row, col].decode('utf-8'))
        else:
            best_a = np.argmax(q_table[s])
            policy.append(actions[best_a])
    
    policy = np.array(policy).reshape(env_desc.shape)
    for row in policy:
        print(" ".join(row))

def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Epsilon starts at 1.0 to encourage exploration at the beginning
    ql_agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0)
    sarsa_agent = SarsaAgent(n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0)
    
    # Train for 15000 episodes to ensure convergence in the stochastic environment
    print("Training Q-Learning Agent...")
    train_agent(env, ql_agent, 15000, is_sarsa=False)
    
    print("Training SARSA Agent...")
    train_agent(env, sarsa_agent, 15000, is_sarsa=True)
    
    ql_successes = []
    sarsa_successes = []
    
    print("Evaluating 100 runs of 100 episodes each...")
    for _ in tqdm(range(100), desc="Evaluations"):
        ql_succ = evaluate_agent(env, ql_agent, 100)
        sarsa_succ = evaluate_agent(env, sarsa_agent, 100)
        ql_successes.append(ql_succ)
        sarsa_successes.append(sarsa_succ)
        
    print("\n================ Resultados FrozenLake ================")
    print(f"Q-Learning Successes: Média = {np.mean(ql_successes):.2f}, Desvio Padrão = {np.std(ql_successes):.2f}")
    print(f"SARSA Successes: Média = {np.mean(sarsa_successes):.2f}, Desvio Padrão = {np.std(sarsa_successes):.2f}")
    
    print("\nPolítica de Ação Preferida - Q-Learning:")
    print_q_table_policy(ql_agent.q_table, env.unwrapped.desc)
    
    print("\nPolítica de Ação Preferida - SARSA:")
    print_q_table_policy(sarsa_agent.q_table, env.unwrapped.desc)
    print("=====================================================")

if __name__ == "__main__":
    main()
