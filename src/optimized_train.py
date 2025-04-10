#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from game_env import Game2048Env
from agent import DQNAgent
from utils import ensure_dir, save_training_metrics, plot_training_results
from tensorflow.keras import optimizers

def train_optimized_model(
    episodes=1000, 
    output_dir='results/optimized',
    target_update_freq=5,
    learning_rate=0.0005,
    batch_size=128,
    gamma=0.995,
    epsilon_decay=0.9998,
    log_freq=50
):
    """Train an optimized model with carefully tuned hyperparameters"""
    # Create output directory
    ensure_dir(output_dir)
    
    # Initialize environment
    env = Game2048Env()
    state_shape = 16  # 4x4 grid flattened
    num_actions = 4   # left, right, up, down
    
    # Create agent with optimized parameters
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Set the custom learning rate
    agent.primary_network.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    agent.target_network.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    # Training metrics
    episode_scores = []
    episode_max_tiles = []
    episode_steps = []
    
    # Save best model state
    best_score = 0
    best_model_path = os.path.join(output_dir, 'best_model.h5')
    
    # Training start time
    start_time = time.time()
    
    print("Starting optimized training...")
    print(f"Parameters: LR={learning_rate}, Batch={batch_size}, Gamma={gamma}, Epsilon decay={epsilon_decay}")
    
    # Training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        steps = 0
        losses = []
        
        while not done:
            # Choose action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            steps += 1
            
            # Store transition in replay buffer
            agent.update_replay_buffer(state, action, reward, next_state, done)
            
            # Train agent
            if agent.replay_buffer.size() > agent.batch_size:
                loss = agent.train()
                losses.append(loss)
            
            # Update target network periodically
            if steps % target_update_freq == 0:
                agent.update_target_network()
            
            # Move to next state
            state = next_state
        
        # Save metrics for this episode
        episode_scores.append(info['score'])
        episode_max_tiles.append(info['max_tile'])
        episode_steps.append(steps)
        
        if losses:
            avg_loss = np.mean(losses)
            agent.loss_history.append(avg_loss)
        
        agent.epsilon_history.append(agent.epsilon)
        
        # Check if this is the best model so far
        if info['score'] > best_score:
            best_score = info['score']
            agent.save_model(best_model_path)
            print(f"New best score: {best_score} - Model saved")
        
        # Log progress
        if episode % log_freq == 0 or episode == episodes:
            avg_score = np.mean(episode_scores[-log_freq:])
            avg_max_tile = np.mean(episode_max_tiles[-log_freq:])
            elapsed_time = time.time() - start_time
            
            print(f"Episode {episode}/{episodes} | "
                  f"Score: {info['score']} | Max Tile: {info['max_tile']} | "
                  f"Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.4f} | "
                  f"Time: {elapsed_time:.2f}s")
        
        # Save and visualize intermediate results every 200 episodes
        if episode % 200 == 0 or episode == episodes:
            # Save the current model
            current_model_path = os.path.join(output_dir, f'model_episode_{episode}.h5')
            agent.save_model(current_model_path)
            
            # Save metrics and plot results
            timestamp = save_training_metrics(agent, episode_scores, episode_max_tiles, output_dir)
            plot_training_results(agent, episode_scores, episode_max_tiles, output_dir, timestamp)
            
            print(f"Saved intermediate results at episode {episode}")
    
    # Final saving
    final_model_path = os.path.join(output_dir, 'final_model.h5')
    agent.save_model(final_model_path)
    
    # Final statistics
    total_time = time.time() - start_time
    final_avg_score = np.mean(episode_scores[-100:])
    max_score = np.max(episode_scores)
    max_tile = np.max(episode_max_tiles)
    
    print("\nTraining Complete!")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Final Average Score (last 100): {final_avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Max Tile: {max_tile}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    
    return best_model_path, final_model_path

def main():
    parser = argparse.ArgumentParser(description='Train an optimized 2048 RL agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--output-dir', type=str, default='results/optimized', help='Directory to save results')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--epsilon-decay', type=float, default=0.9998, help='Epsilon decay rate')
    parser.add_argument('--log-freq', type=int, default=50, help='Frequency of progress logging')
    
    args = parser.parse_args()
    
    train_optimized_model(
        episodes=args.episodes,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        log_freq=args.log_freq
    )

if __name__ == "__main__":
    main() 