import os
import numpy as np
import argparse
import time
from game_env import Game2048Env
from agent import DQNAgent
from utils import visualize_game_state, create_game_gif

def evaluate_agent(agent, env, num_episodes=10, render=True, output_dir='results/evaluation',
                  save_game_states=True, create_gif=True):
    """Evaluate the agent's performance with no exploration"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporarily set epsilon to 0 for evaluation (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    scores = []
    max_tiles = []
    steps_list = []
    all_game_images = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        episode_images = []
        
        # For tracking game state for visualization
        if save_game_states:
            episode_timestamp = time.strftime("%Y%m%d-%H%M%S")
            episode_dir = os.path.join(output_dir, f'episode_{episode}_{episode_timestamp}')
            os.makedirs(episode_dir, exist_ok=True)
            
            # Save initial state
            img_path = visualize_game_state(
                state, env.score, np.max(state),
                output_dir=episode_dir,
                filename=f'step_0.png'
            )
            episode_images.append(img_path)
        
        if render:
            print(f"\nEvaluation Episode {episode + 1}/{num_episodes}")
            env.render()
        
        while not done:
            # Choose the best action according to the learned policy
            action = agent.get_action(state)  # With epsilon=0, this will always choose the best action
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Save game state for visualization
            if save_game_states:
                img_path = visualize_game_state(
                    next_state, info['score'], info['max_tile'],
                    action_name=Game2048Env.ACTIONS[action],
                    output_dir=episode_dir,
                    filename=f'step_{steps}.png'
                )
                episode_images.append(img_path)
            
            if render:
                print(f"\nAction: {Game2048Env.ACTIONS[action]}")
                env.render()
            
            # Move to the next state
            state = next_state
        
        # Store episode results
        scores.append(info['score'])
        max_tiles.append(info['max_tile'])
        steps_list.append(steps)
        
        # Create GIF for this episode
        if create_gif and save_game_states and len(episode_images) > 1:
            gif_path = os.path.join(episode_dir, f'episode_{episode}_animation.gif')
            create_game_gif(episode_images, gif_path, duration=0.3)
            all_game_images.append(episode_images)
        
        if render:
            print(f"Episode {episode + 1} Results:")
            print(f"Score: {info['score']}")
            print(f"Max Tile: {info['max_tile']}")
            print(f"Steps: {steps}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate statistics
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    avg_max_tile = np.mean(max_tiles)
    max_tile = np.max(max_tiles)
    avg_steps = np.mean(steps_list)
    
    # Print evaluation results
    print(f"\nEvaluation Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Average Max Tile: {avg_max_tile:.2f}")
    print(f"Max Tile Achieved: {max_tile}")
    print(f"Average Steps: {avg_steps:.2f}")
    
    # Save results to file
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results:\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Average Score: {avg_score:.2f}\n")
        f.write(f"Max Score: {max_score}\n")
        f.write(f"Average Max Tile: {avg_max_tile:.2f}\n")
        f.write(f"Max Tile Achieved: {max_tile}\n")
        f.write(f"Average Steps: {avg_steps:.2f}\n")
        
        # Add detailed results for each episode
        f.write("\nDetailed Results:\n")
        for ep in range(num_episodes):
            f.write(f"Episode {ep + 1}: Score={scores[ep]}, Max Tile={max_tiles[ep]}, Steps={steps_list[ep]}\n")
    
    return scores, max_tiles, steps_list, all_game_images

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN agent on 2048')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/evaluation', help='Directory to save results')
    parser.add_argument('--no-render', action='store_true', help='Disable console rendering')
    parser.add_argument('--no-save-states', action='store_true', help='Disable saving game states')
    parser.add_argument('--no-gif', action='store_true', help='Disable GIF creation')
    
    args = parser.parse_args()
    
    # Initialize environment and agent
    env = Game2048Env()
    state_shape = 16  # 4x4 grid flattened
    num_actions = 4   # left, right, up, down
    
    # Create agent
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions
    )
    
    # Load trained model
    try:
        agent.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate the agent
    print(f"Evaluating agent for {args.episodes} episodes...")
    
    evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        render=not args.no_render,
        output_dir=args.output_dir,
        save_game_states=not args.no_save_states,
        create_gif=not args.no_gif
    )

if __name__ == "__main__":
    main() 