import os
import numpy as np
import time
import argparse
from game_env import Game2048Env
from agent import DQNAgent
from utils import save_training_metrics, plot_training_results, visualize_game_state

def train_agent(agent, env, num_episodes=5000, target_update_freq=10, 
                visualize_freq=500, save_freq=100, log_freq=100, output_dir='results',
                save_game_states=False, state_save_freq=1000, max_state_saves=50):
    """Train the agent for a specified number of episodes"""
    max_score = 0
    max_tile = 0
    episode_scores = []
    episode_max_tiles = []
    game_state_images = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Training start time
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # For visualizing this episode's game states
        if save_game_states and (episode % state_save_freq == 0) and len(game_state_images) < max_state_saves:
            episode_images = []
            episode_timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save initial state
            initial_img = visualize_game_state(
                state, env.score, np.max(state), 
                output_dir=os.path.join(output_dir, 'game_states'),
                filename=f'episode_{episode}_step_0_{episode_timestamp}.png'
            )
            episode_images.append(initial_img)
        else:
            episode_images = None
        
        while not done:
            # Choose an action
            action = agent.get_action(state)
            
            # Take a step in the environment
            next_state, reward, done, info = env.step(action)
            
            # Save game state for visualization
            if episode_images is not None and (steps < 50 or steps % 5 == 0):  # Limit number of saved states
                img = visualize_game_state(
                    next_state, info['score'], info['max_tile'], 
                    action_name=Game2048Env.ACTIONS[action],
                    output_dir=os.path.join(output_dir, 'game_states'),
                    filename=f'episode_{episode}_step_{steps+1}_{episode_timestamp}.png'
                )
                episode_images.append(img)
            
            # Store the experience in replay buffer
            agent.update_replay_buffer(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train()
            if loss > 0:
                agent.loss_history.append(loss)
            
            # Move to the next state
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Update the target network periodically
            if steps % target_update_freq == 0:
                agent.update_target_network()
        
        # Store episode results
        episode_scores.append(info['score'])
        episode_max_tiles.append(info['max_tile'])
        
        # Update max score and max tile
        if info['score'] > max_score:
            max_score = info['score']
        if info['max_tile'] > max_tile:
            max_tile = info['max_tile']
        
        # Track epsilon
        agent.epsilon_history.append(agent.epsilon)
        
        # Log progress periodically
        if (episode + 1) % log_freq == 0:
            elapsed_time = time.time() - start_time
            avg_score = np.mean(episode_scores[-log_freq:])
            avg_max_tile = np.mean(episode_max_tiles[-log_freq:])
            print(f"Episode {episode + 1}/{num_episodes} | " +
                  f"Avg Score: {avg_score:.1f} | " +
                  f"Avg Max Tile: {avg_max_tile:.1f} | " +
                  f"Epsilon: {agent.epsilon:.4f} | " +
                  f"Time: {elapsed_time:.1f}s")
        
        # Visualize/save results periodically
        if (episode + 1) % visualize_freq == 0 or (episode + 1) == num_episodes:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Score: {info['score']}, Max Tile: {info['max_tile']}, Epsilon: {agent.epsilon:.4f}")
            print(f"Max Score so far: {max_score}, Max Tile so far: {max_tile}")
            
            # Save metrics and generate plots
            timestamp = save_training_metrics(agent, episode_scores, episode_max_tiles, output_dir)
            plot_training_results(agent, episode_scores, episode_max_tiles, output_dir, timestamp)
        
        # Save the model periodically
        if (episode + 1) % save_freq == 0 or (episode + 1) == num_episodes:
            model_dir = os.path.join(output_dir, 'models')
            agent.save_model(os.path.join(model_dir, f"dqn_2048_episode_{episode + 1}.h5"))
            
            # Save final model with a simpler name
            if (episode + 1) == num_episodes:
                agent.save_model(os.path.join(model_dir, "dqn_2048_final.h5"))
        
        # Save game state images for this episode if collected
        if episode_images and len(episode_images) > 1:
            game_state_images.append((episode, episode_images))
    
    # Final training stats
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total training time: {total_time:.1f} seconds")
    print(f"Max Score: {max_score}")
    print(f"Max Tile: {max_tile}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    
    return episode_scores, episode_max_tiles, game_state_images

def main():
    parser = argparse.ArgumentParser(description='Train a DQN agent to play 2048')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train for')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--target-update-freq', type=int, default=10, help='Frequency of target network updates')
    parser.add_argument('--visualize-freq', type=int, default=500, help='Frequency of visualization updates')
    parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of model saving')
    parser.add_argument('--log-freq', type=int, default=100, help='Frequency of progress logging')
    parser.add_argument('--save-game-states', action='store_true', help='Save game state visualizations')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.9995, help='Exploration decay rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Initialize environment and agent
    env = Game2048Env()
    state_shape = 16  # 4x4 grid flattened
    num_actions = 4   # left, right, up, down
    
    # Create agent with specified hyperparameters
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size
    )
    
    # Train the agent
    print(f"Starting training for {args.episodes} episodes...")
    
    episode_scores, episode_max_tiles, game_state_images = train_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        target_update_freq=args.target_update_freq,
        visualize_freq=args.visualize_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        output_dir=args.output_dir,
        save_game_states=args.save_game_states
    )

if __name__ == "__main__":
    main() 