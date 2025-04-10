import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='2048 Reinforcement Learning Agent')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train for')
    train_parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    train_parser.add_argument('--target-update-freq', type=int, default=10, help='Frequency of target network updates')
    train_parser.add_argument('--visualize-freq', type=int, default=500, help='Frequency of visualization updates')
    train_parser.add_argument('--save-freq', type=int, default=1000, help='Frequency of model saving')
    train_parser.add_argument('--log-freq', type=int, default=100, help='Frequency of progress logging')
    train_parser.add_argument('--save-game-states', action='store_true', help='Save game state visualizations')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    train_parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    train_parser.add_argument('--epsilon-decay', type=float, default=0.9995, help='Exploration decay rate')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained agent')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')
    eval_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    eval_parser.add_argument('--output-dir', type=str, default='results/evaluation', help='Directory to save results')
    eval_parser.add_argument('--no-render', action='store_true', help='Disable console rendering')
    eval_parser.add_argument('--no-save-states', action='store_true', help='Disable saving game states')
    eval_parser.add_argument('--no-gif', action='store_true', help='Disable GIF creation')
    
    # Test command (play a single game with rendering)
    test_parser = subparsers.add_parser('test', help='Play a single game with the agent')
    test_parser.add_argument('--model-path', type=str, help='Path to the trained model weights')
    test_parser.add_argument('--random', action='store_true', help='Use random actions instead of the model')
    test_parser.add_argument('--output-dir', type=str, default='results/test', help='Directory to save results')
    
    # Decision visualization command
    viz_parser = subparsers.add_parser('visualize-decisions', help='Visualize the decision-making process of the agent')
    viz_parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')
    viz_parser.add_argument('--output-dir', type=str, default='results/decision_viz', help='Directory to save visualizations')
    viz_parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of steps to run')
    viz_parser.add_argument('--visualize-only', action='store_true', help='Only create visualizations without GIF')
    
    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser('tune', help='Tune hyperparameters for the agent')
    tune_parser.add_argument('--episodes', type=int, default=200, help='Number of episodes per configuration')
    tune_parser.add_argument('--output-dir', type=str, default='results/hyperparameter_tuning', help='Directory to save results')
    tune_parser.add_argument('--configs', type=str, default='all', help='Comma-separated list of configurations to test (or "all")')
    tune_parser.add_argument('--log-freq', type=int, default=50, help='Frequency of progress logging')
    
    # Quick hyperparameter tuning command
    quick_tune_parser = subparsers.add_parser('quick-tune', help='Quickly test essential hyperparameters')
    quick_tune_parser.add_argument('--episodes', type=int, default=50, help='Number of episodes per configuration')
    quick_tune_parser.add_argument('--output-dir', type=str, default='results/quick_tune', help='Directory to save results')
    
    # Optimized training command
    optimized_train_parser = subparsers.add_parser('optimized-train', help='Train model with optimized hyperparameters')
    optimized_train_parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train for')
    optimized_train_parser.add_argument('--output-dir', type=str, default='results/optimized', help='Directory to save results')
    optimized_train_parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate for the optimizer')
    optimized_train_parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    optimized_train_parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    optimized_train_parser.add_argument('--epsilon-decay', type=float, default=0.9998, help='Epsilon decay rate')
    optimized_train_parser.add_argument('--log-freq', type=int, default=50, help='Frequency of progress logging')
    
    # Play command for live visualization
    play_parser = subparsers.add_parser('play', help='Watch the agent play in real-time')
    play_parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')
    play_parser.add_argument('--delay', type=float, default=0.5, help='Delay between moves in seconds')
    play_parser.add_argument('--save-dir', type=str, default=None, help='Directory to save screenshots')
    play_parser.add_argument('--max-steps', type=int, default=200, help='Maximum number of steps')
    play_parser.add_argument('--hide-q-values', action='store_true', help='Hide Q-values visualization')
    
    args = parser.parse_args()
    
    # Add the src directory to the path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.command == 'train':
        from train import main as train_main
        # Reconstruct sys.argv for the train script
        sys.argv = ['train.py']
        if args.episodes != 5000:
            sys.argv.extend(['--episodes', str(args.episodes)])
        if args.output_dir != 'results':
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.target_update_freq != 10:
            sys.argv.extend(['--target-update-freq', str(args.target_update_freq)])
        if args.visualize_freq != 500:
            sys.argv.extend(['--visualize-freq', str(args.visualize_freq)])
        if args.save_freq != 1000:
            sys.argv.extend(['--save-freq', str(args.save_freq)])
        if args.log_freq != 100:
            sys.argv.extend(['--log-freq', str(args.log_freq)])
        if args.save_game_states:
            sys.argv.append('--save-game-states')
        if args.gamma != 0.99:
            sys.argv.extend(['--gamma', str(args.gamma)])
        if args.epsilon != 1.0:
            sys.argv.extend(['--epsilon', str(args.epsilon)])
        if args.epsilon_min != 0.01:
            sys.argv.extend(['--epsilon-min', str(args.epsilon_min)])
        if args.epsilon_decay != 0.9995:
            sys.argv.extend(['--epsilon-decay', str(args.epsilon_decay)])
        if args.batch_size != 64:
            sys.argv.extend(['--batch-size', str(args.batch_size)])
        
        train_main()
    
    elif args.command == 'evaluate':
        from evaluate import main as evaluate_main
        # Reconstruct sys.argv for the evaluate script
        sys.argv = ['evaluate.py', '--model-path', args.model_path]
        if args.episodes != 10:
            sys.argv.extend(['--episodes', str(args.episodes)])
        if args.output_dir != 'results/evaluation':
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.no_render:
            sys.argv.append('--no-render')
        if args.no_save_states:
            sys.argv.append('--no-save-states')
        if args.no_gif:
            sys.argv.append('--no-gif')
        
        evaluate_main()
    
    elif args.command == 'test':
        # Create a simple test function to play a single game
        from game_env import Game2048Env
        from agent import DQNAgent
        from utils import visualize_game_state, create_game_gif
        import time
        import numpy as np
        
        # Initialize environment
        env = Game2048Env()
        state_shape = 16  # 4x4 grid flattened
        num_actions = 4   # left, right, up, down
        
        # Create agent
        agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions
        )
        
        # Load model if specified
        if args.model_path and not args.random:
            try:
                agent.load_model(args.model_path)
                print(f"Loaded model from {args.model_path}")
                # Set epsilon to 0 for deterministic behavior
                agent.epsilon = 0.0
            except Exception as e:
                print(f"Error loading model: {e}")
                return
        elif args.random:
            print("Using random actions for testing")
            agent.epsilon = 1.0  # Always use random actions
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Play a single game
        state = env.reset()
        done = False
        steps = 0
        episode_images = []
        episode_timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save initial state
        img_path = visualize_game_state(
            state, env.score, np.max(state),
            output_dir=args.output_dir,
            filename=f'step_0_{episode_timestamp}.png'
        )
        episode_images.append(img_path)
        
        print("\n*** Starting Test Game ***")
        env.render()
        
        while not done:
            # Choose action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            steps += 1
            
            # Save game state
            img_path = visualize_game_state(
                next_state, info['score'], info['max_tile'],
                action_name=Game2048Env.ACTIONS[action],
                output_dir=args.output_dir,
                filename=f'step_{steps}_{episode_timestamp}.png'
            )
            episode_images.append(img_path)
            
            print(f"\nAction: {Game2048Env.ACTIONS[action]}")
            env.render()
            
            # Move to next state
            state = next_state
        
        # Create GIF
        gif_path = os.path.join(args.output_dir, f'game_animation_{episode_timestamp}.gif')
        create_game_gif(episode_images, gif_path, duration=0.5)
        
        # Show results
        print("\n*** Test Game Results ***")
        print(f"Score: {info['score']}")
        print(f"Max Tile: {info['max_tile']}")
        print(f"Steps: {steps}")
        print(f"Game states saved to: {args.output_dir}")
        print(f"Game animation: {gif_path}")
    
    elif args.command == 'visualize-decisions':
        from visualize_decisions import main as viz_main
        # Reconstruct sys.argv for the visualize_decisions script
        sys.argv = ['visualize_decisions.py', '--model-path', args.model_path]
        if args.output_dir != 'results/decision_viz':
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.max_steps != 100:
            sys.argv.extend(['--max-steps', str(args.max_steps)])
        if args.visualize_only:
            sys.argv.append('--visualize-only')
        
        viz_main()
    
    elif args.command == 'tune':
        from hyperparameter_tuning import main as tune_main
        # Reconstruct sys.argv for the tuning script
        sys.argv = ['hyperparameter_tuning.py']
        if args.episodes != 200:
            sys.argv.extend(['--episodes', str(args.episodes)])
        if args.output_dir != 'results/hyperparameter_tuning':
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.configs != 'all':
            sys.argv.extend(['--configs', args.configs])
        if args.log_freq != 50:
            sys.argv.extend(['--log-freq', str(args.log_freq)])
        
        tune_main()
    
    elif args.command == 'quick-tune':
        from quick_tune import main as quick_tune_main
        # Reconstruct sys.argv for the quick tuning script
        sys.argv = ['quick_tune.py']
        if args.episodes != 50:
            sys.argv.extend(['--episodes', str(args.episodes)])
        if args.output_dir != 'results/quick_tune':
            sys.argv.extend(['--output-dir', args.output_dir])
        
        quick_tune_main()
    
    elif args.command == 'optimized-train':
        from optimized_train import main as optimized_train_main
        # Reconstruct sys.argv for the optimized training script
        sys.argv = ['optimized_train.py']
        if args.episodes != 1000:
            sys.argv.extend(['--episodes', str(args.episodes)])
        if args.output_dir != 'results/optimized':
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.learning_rate != 0.0005:
            sys.argv.extend(['--learning-rate', str(args.learning_rate)])
        if args.batch_size != 128:
            sys.argv.extend(['--batch-size', str(args.batch_size)])
        if args.gamma != 0.995:
            sys.argv.extend(['--gamma', str(args.gamma)])
        if args.epsilon_decay != 0.9998:
            sys.argv.extend(['--epsilon-decay', str(args.epsilon_decay)])
        if args.log_freq != 50:
            sys.argv.extend(['--log-freq', str(args.log_freq)])
        
        optimized_train_main()
    
    elif args.command == 'play':
        from live_play import main as live_play_main
        # Reconstruct sys.argv for the live play script
        sys.argv = ['live_play.py', '--model-path', args.model_path]
        if args.delay != 0.5:
            sys.argv.extend(['--delay', str(args.delay)])
        if args.save_dir:
            sys.argv.extend(['--save-dir', args.save_dir])
        if args.max_steps != 200:
            sys.argv.extend(['--max-steps', str(args.max_steps)])
        if args.hide_q_values:
            sys.argv.append('--hide-q-values')
        
        live_play_main()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 