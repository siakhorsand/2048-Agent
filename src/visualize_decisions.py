#!/usr/bin/env python3
import os
import argparse
from game_env import Game2048Env
from agent import DQNAgent
from decision_viz import visualize_q_values, create_decision_making_animation

def main():
    """
    Demonstrates the decision-making visualization for a trained 2048 agent.
    """
    parser = argparse.ArgumentParser(description='Visualize decision-making process of a trained 2048 agent')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--output-dir', type=str, default='results/decision_viz', help='Directory to save visualizations')
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps to run')
    parser.add_argument('--visualize-only', action='store_true', help='Only create visualizations without GIF')
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run decision visualization
    print("Creating decision-making visualizations...")
    
    if args.visualize_only:
        # Just create a visualization for the initial state
        state = env.reset()
        output_path = visualize_q_values(
            agent, state, 
            output_dir=args.output_dir,
            filename='initial_state_decision.png'
        )
        print(f"Created visualization: {output_path}")
    else:
        # Create a full animation showing decision-making over time
        gif_path, _ = create_decision_making_animation(
            agent, env, 
            max_steps=args.max_steps,
            output_dir=args.output_dir
        )
        
        if gif_path:
            print(f"Successfully created decision visualization at {gif_path}")
        else:
            print("Failed to create GIF animation, but individual frames were saved.")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 