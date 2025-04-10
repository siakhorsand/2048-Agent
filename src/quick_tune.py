#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from game_env import Game2048Env
from agent import DQNAgent, create_dqn_model, preprocess_state
from utils import ensure_dir
from tensorflow.keras import layers, models, optimizers

def test_config(config, episodes=50, output_dir='results/quick_tune'):
    """Test a single configuration with a small number of episodes"""
    config_name = config['name']
    print(f"\nTesting configuration: {config_name}")
    print('-' * 50)
    for key, value in config.items():
        if key != 'name':
            print(f"{key}: {value}")
    
    # Create output directory for this configuration
    config_dir = os.path.join(output_dir, config_name)
    ensure_dir(config_dir)
    
    # Initialize environment and agent
    env = Game2048Env()
    state_shape = 16  # 4x4 grid flattened
    num_actions = 4   # left, right, up, down
    
    # Create agent with custom parameters
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        gamma=config.get('gamma', 0.99),
        epsilon=config.get('epsilon', 1.0),
        epsilon_min=config.get('epsilon_min', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.9995),
        batch_size=config.get('batch_size', 64)
    )
    
    # If learning rate is specified, update the model optimizer
    if 'learning_rate' in config:
        agent.primary_network.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']), 
            loss='mse'
        )
        agent.target_network.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']), 
            loss='mse'
        )
    
    # Training metrics
    episode_scores = []
    episode_max_tiles = []
    start_time = time.time()
    
    # Run short training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            # Choose action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            steps += 1
            
            # Store transition
            agent.update_replay_buffer(state, action, reward, next_state, done)
            
            # Train agent
            if agent.replay_buffer.size() > agent.batch_size:
                agent.train()
            
            # Update target network every 10 steps
            if steps % 10 == 0:
                agent.update_target_network()
            
            # Move to next state
            state = next_state
        
        # Save metrics
        episode_scores.append(info['score'])
        episode_max_tiles.append(info['max_tile'])
        
        # Log progress
        if episode % 10 == 0 or episode == episodes:
            avg_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode}/{episodes} | Score: {info['score']} | "
                  f"Max Tile: {info['max_tile']} | Avg Score: {avg_score:.2f}")
    
    # Calculate final metrics
    train_time = time.time() - start_time
    avg_score = np.mean(episode_scores)
    max_tile = np.max(episode_max_tiles)
    
    # Create simple score plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_scores)
    plt.title(f'Scores for {config_name}')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config_dir, 'scores.png'), dpi=150)
    plt.close()
    
    return {
        'config_name': config_name,
        'avg_score': avg_score,
        'max_tile': max_tile,
        'train_time': train_time
    }

def quick_tune(episodes=50, output_dir='results/quick_tune'):
    """Run a quick hyperparameter test with the most important parameters"""
    ensure_dir(output_dir)
    
    # Define a small set of configurations focusing on key parameters
    configs = [
        {
            'name': 'baseline',
            'gamma': 0.99,
            'epsilon_decay': 0.9995,
            'learning_rate': 0.0001,
            'batch_size': 64
        },
        {
            'name': 'high_lr',
            'learning_rate': 0.001
        },
        {
            'name': 'low_lr',
            'learning_rate': 0.00001
        },
        {
            'name': 'fast_explore',
            'epsilon_decay': 0.999
        },
        {
            'name': 'slow_explore',
            'epsilon_decay': 0.9999
        }
    ]
    
    # Run tests
    results = []
    for config in configs:
        result = test_config(config, episodes, output_dir)
        results.append(result)
    
    # Compare results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_score', ascending=False)
    
    # Save comparison to CSV
    results_df.to_csv(os.path.join(output_dir, 'quick_tune_results.csv'), index=False)
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(results_df['config_name'], results_df['avg_score'], color='skyblue')
    plt.title('Average Score by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_comparison.png'), dpi=150)
    plt.close()
    
    # Print results
    print("\nQuick Tuning Results (Best to Worst):")
    print("=" * 60)
    for _, row in results_df.iterrows():
        print(f"Config: {row['config_name']}")
        print(f"  Avg Score: {row['avg_score']:.2f}, Max Tile: {row['max_tile']}")
        print(f"  Training Time: {row['train_time']:.2f}s")
        print("-" * 60)
    
    # Return best configuration
    best_config = results_df.iloc[0]['config_name']
    print(f"\nBest configuration: {best_config}")
    print(f"Recommended for full training.")
    
    return best_config

def main():
    parser = argparse.ArgumentParser(description='Quick hyperparameter tuning for 2048 RL agent')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes per configuration')
    parser.add_argument('--output-dir', type=str, default='results/quick_tune', help='Directory to save results')
    
    args = parser.parse_args()
    
    print("Starting quick hyperparameter tuning...")
    print(f"Testing 5 configurations with {args.episodes} episodes each")
    
    best_config = quick_tune(args.episodes, args.output_dir)
    
    print(f"\nQuick tuning complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 