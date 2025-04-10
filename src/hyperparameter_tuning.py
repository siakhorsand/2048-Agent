#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from game_env import Game2048Env
from agent import DQNAgent, create_dqn_model
from utils import ensure_dir
from tensorflow.keras import layers, models, optimizers

def preprocess_state(state):
    """Preprocess the game state for neural network input"""
    # Normalize the state by taking log2 of non-zero values (since tiles are powers of 2)
    # This helps the neural network learn more efficiently
    processed = np.zeros_like(state, dtype=np.float32)
    
    # For each non-zero tile, take log2 and normalize by dividing by 16 (max tile is usually 2^15 = 32768)
    non_zero_mask = state > 0
    processed[non_zero_mask] = np.log2(state[non_zero_mask]) / 16.0
    
    # Reshape to (1, 16) for Dense layers
    return processed.reshape(1, -1)

def create_custom_model(input_shape, num_actions, architecture, learning_rate=0.0001):
    """Create a custom DQN model based on the specified architecture"""
    model = models.Sequential()
    
    # Add input layer
    model.add(layers.Dense(architecture[0], activation='relu', input_shape=input_shape))
    
    # Add hidden layers
    for units in architecture[1:]:
        model.add(layers.Dense(units, activation='relu'))
    
    # Add output layer
    model.add(layers.Dense(num_actions, activation='linear'))
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

def test_hyperparameters(config, episodes=200, log_freq=50, output_dir='results/hyperparameter_tuning'):
    """Test a specific hyperparameter configuration"""
    config_name = config['name']
    print(f"\nTesting configuration: {config_name}")
    print('-' * 50)
    for key, value in config.items():
        if key != 'name':
            print(f"{key}: {value}")
    
    # Create output directory for this configuration
    config_dir = os.path.join(output_dir, config_name)
    ensure_dir(config_dir)
    
    # Initialize environment
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
    
    # If a custom architecture is specified, create and use that model
    architecture = config.get('architecture', None)
    learning_rate = config.get('learning_rate', 0.0001)
    
    if architecture:
        agent.primary_network = create_custom_model(
            (state_shape,), 
            num_actions, 
            architecture, 
            learning_rate
        )
        agent.target_network = create_custom_model(
            (state_shape,), 
            num_actions, 
            architecture, 
            learning_rate
        )
        agent.update_target_network()
    
    # Training metrics
    episode_scores = []
    episode_max_tiles = []
    episode_losses = []
    episode_steps = []
    
    # Training loop
    start_time = time.time()
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        losses = []
        steps = 0
        done = False
        
        while not done:
            # Select action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Store transition in replay buffer
            agent.update_replay_buffer(state, action, reward, next_state, done)
            
            # Train agent
            if agent.replay_buffer.size() > agent.batch_size:
                loss = agent.train()
                losses.append(loss)
            
            # Move to next state
            state = next_state
            
            # Update target network periodically
            if steps % config.get('target_update_freq', 10) == 0:
                agent.update_target_network()
        
        # Save metrics for this episode
        episode_scores.append(info['score'])
        episode_max_tiles.append(info['max_tile'])
        episode_steps.append(steps)
        if losses:
            episode_losses.append(np.mean(losses))
            agent.loss_history.append(np.mean(losses))
        agent.epsilon_history.append(agent.epsilon)
        
        # Log progress
        if episode % log_freq == 0 or episode == episodes:
            avg_score = np.mean(episode_scores[-log_freq:])
            avg_max_tile = np.mean(episode_max_tiles[-log_freq:])
            current_time = time.time() - start_time
            print(f"Episode {episode}/{episodes} | " 
                  f"Score: {info['score']} | Max Tile: {info['max_tile']} | "
                  f"Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {current_time:.2f}s")
    
    # Calculate final statistics
    train_time = time.time() - start_time
    final_avg_score = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
    max_score = np.max(episode_scores)
    max_tile = np.max(episode_max_tiles)
    avg_steps = np.mean(episode_steps)
    
    # Save results
    results = {
        'config_name': config_name,
        'episodes': episodes,
        'final_avg_score': final_avg_score,
        'max_score': max_score,
        'max_tile': max_tile,
        'avg_steps': avg_steps,
        'train_time': train_time
    }
    
    # Add configuration parameters to results
    for key, value in config.items():
        if key != 'name' and key != 'architecture':
            results[key] = value
    
    # Convert architecture to string if present
    if 'architecture' in config:
        results['architecture'] = str(config['architecture'])
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'score': episode_scores,
        'max_tile': episode_max_tiles,
        'epsilon': agent.epsilon_history,
        'steps': episode_steps
    })
    metrics_df.to_csv(os.path.join(config_dir, 'training_metrics.csv'), index=False)
    
    # Save losses if available
    if episode_losses:
        pd.DataFrame({'loss': episode_losses}).to_csv(
            os.path.join(config_dir, 'loss_history.csv'), index=False)
    
    # Plot learning curves
    plot_learning_curves(episode_scores, episode_max_tiles, 
                         agent.loss_history, agent.epsilon_history, 
                         config_name, config_dir)
    
    return results

def plot_learning_curves(scores, max_tiles, losses, epsilons, config_name, output_dir):
    """Plot and save the learning curves for the hyperparameter configuration"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot scores
    axs[0, 0].plot(scores)
    axs[0, 0].set_title('Score per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Score')
    
    # Plot max tiles
    axs[0, 1].plot(max_tiles)
    axs[0, 1].set_title('Max Tile per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Max Tile')
    
    # Plot losses if available
    if losses:
        axs[1, 0].plot(losses)
        axs[1, 0].set_title('Loss over Training Steps')
        axs[1, 0].set_xlabel('Training Step')
        axs[1, 0].set_ylabel('Loss')
    
    # Plot epsilon
    if epsilons:
        axs[1, 1].plot(epsilons)
        axs[1, 1].set_title('Epsilon over Episodes')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Epsilon')
    
    plt.tight_layout()
    plt.suptitle(f'Learning Curves: {config_name}', y=1.02)
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_results(results, output_dir='results/hyperparameter_tuning'):
    """Compare the results of different hyperparameter configurations"""
    if not results:
        print("No results to compare")
        return
    
    # Convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Save comparison to CSV
    results_df.to_csv(os.path.join(output_dir, 'hyperparameter_comparison.csv'), index=False)
    
    # Create bar charts for key metrics
    metrics = ['final_avg_score', 'max_score', 'max_tile', 'train_time']
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        results_df.plot.bar(x='config_name', y=metric, ax=axs[row, col], legend=False)
        axs[row, col].set_title(f'Comparison: {metric}')
        axs[row, col].set_ylabel(metric)
        axs[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Print results sorted by final average score
    print("\nResults Summary (Sorted by Final Avg Score):")
    print("=" * 80)
    sorted_results = results_df.sort_values('final_avg_score', ascending=False)
    for _, row in sorted_results.iterrows():
        print(f"Config: {row['config_name']}")
        print(f"  Avg Score: {row['final_avg_score']:.2f}, Max Score: {row['max_score']}")
        print(f"  Max Tile: {row['max_tile']}, Train Time: {row['train_time']:.2f}s")
        print("-" * 80)

def default_configurations():
    """Define a set of default hyperparameter configurations to test"""
    configs = []
    
    # Baseline configuration
    configs.append({
        'name': 'baseline',
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.9995,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'architecture': [256, 256, 128],
        'target_update_freq': 10
    })
    
    # Network architecture variations
    configs.append({
        'name': 'wider_network',
        'architecture': [512, 512, 256],
        'learning_rate': 0.0001
    })
    
    configs.append({
        'name': 'deeper_network',
        'architecture': [256, 256, 256, 128],
        'learning_rate': 0.0001
    })
    
    # Learning rate variations
    configs.append({
        'name': 'higher_lr',
        'learning_rate': 0.001
    })
    
    configs.append({
        'name': 'lower_lr',
        'learning_rate': 0.00001
    })
    
    # Exploration/exploitation balance
    configs.append({
        'name': 'faster_exploration',
        'epsilon_decay': 0.999
    })
    
    configs.append({
        'name': 'slower_exploration',
        'epsilon_decay': 0.9999
    })
    
    # Discount factor variations
    configs.append({
        'name': 'higher_gamma',
        'gamma': 0.995
    })
    
    configs.append({
        'name': 'lower_gamma',
        'gamma': 0.95
    })
    
    # Batch size variations
    configs.append({
        'name': 'larger_batch',
        'batch_size': 128
    })
    
    configs.append({
        'name': 'smaller_batch',
        'batch_size': 32
    })
    
    # Target network update frequency
    configs.append({
        'name': 'frequent_target_update',
        'target_update_freq': 5
    })
    
    configs.append({
        'name': 'infrequent_target_update',
        'target_update_freq': 20
    })
    
    return configs

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for 2048 RL agent')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes per configuration')
    parser.add_argument('--output-dir', type=str, default='results/hyperparameter_tuning',
                        help='Directory to save results')
    parser.add_argument('--configs', type=str, default='all',
                        help='Comma-separated list of configurations to test (or "all")')
    parser.add_argument('--log-freq', type=int, default=50, help='Frequency of progress logging')
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Get configurations to test
    all_configs = default_configurations()
    
    if args.configs.lower() == 'all':
        configs_to_test = all_configs
    else:
        config_names = args.configs.split(',')
        configs_to_test = [config for config in all_configs if config['name'] in config_names]
        
        if not configs_to_test:
            print(f"No valid configurations found. Available configurations:")
            for config in all_configs:
                print(f"  - {config['name']}")
            return
    
    # Print configurations to test
    print(f"Testing {len(configs_to_test)} configurations with {args.episodes} episodes each:")
    for config in configs_to_test:
        print(f"  - {config['name']}")
    
    # Test each configuration
    results = []
    for config in configs_to_test:
        result = test_hyperparameters(
            config, 
            episodes=args.episodes,
            log_freq=args.log_freq,
            output_dir=args.output_dir
        )
        results.append(result)
    
    # Compare results
    compare_results(results, args.output_dir)
    
    print(f"\nHyperparameter tuning complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 