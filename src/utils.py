import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

def ensure_dir(directory):
    """Make sure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_training_metrics(agent, episode_scores, episode_max_tiles, output_dir='results'):
    """Save training metrics as CSV files"""
    ensure_dir(output_dir)
    
    # Create timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save loss history
    if len(agent.loss_history) > 0:
        pd.DataFrame({'loss': agent.loss_history}).to_csv(
            os.path.join(output_dir, f'loss_history_{timestamp}.csv'), index=False)
    
    # Save scores and max tiles
    metrics_df = pd.DataFrame({
        'episode': range(1, len(episode_scores) + 1),
        'score': episode_scores,
        'max_tile': episode_max_tiles,
        'epsilon': agent.epsilon_history if len(agent.epsilon_history) == len(episode_scores) else [None] * len(episode_scores)
    })
    metrics_df.to_csv(os.path.join(output_dir, f'training_metrics_{timestamp}.csv'), index=False)
    
    return timestamp

def plot_training_results(agent, episode_scores, episode_max_tiles, output_dir='results', timestamp=None):
    """Plot and save the training results as PNG files"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    ensure_dir(output_dir)
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot the score over episodes
    axs[0, 0].plot(episode_scores)
    axs[0, 0].set_title('Score per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Score')
    
    # Plot the max tile over episodes
    axs[0, 1].plot(episode_max_tiles)
    axs[0, 1].set_title('Max Tile per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Max Tile')
    
    # Plot the loss over training steps
    if len(agent.loss_history) > 0:
        axs[1, 0].plot(agent.loss_history)
        axs[1, 0].set_title('Loss over Training Steps')
        axs[1, 0].set_xlabel('Training Step')
        axs[1, 0].set_ylabel('Loss')
    
    # Plot the epsilon over episodes
    if len(agent.epsilon_history) > 0:
        axs[1, 1].plot(agent.epsilon_history)
        axs[1, 1].set_title('Epsilon over Episodes')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Epsilon')
    
    # Calculate moving averages for smoother plots
    window_size = min(100, len(episode_scores))
    if window_size > 0 and len(episode_scores) > window_size:
        scores_avg = np.convolve(episode_scores, np.ones(window_size)/window_size, mode='valid')
        axs[0, 0].plot(range(window_size-1, len(scores_avg) + window_size-1), scores_avg, 'r-', label=f'{window_size}-episode avg')
        axs[0, 0].legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'training_results_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create individual plots with higher quality for each metric
    create_single_plot(episode_scores, 'Score per Episode', 'Episode', 'Score', 
                      os.path.join(output_dir, f'scores_{timestamp}.png'), window_size)
    
    create_single_plot(episode_max_tiles, 'Max Tile per Episode', 'Episode', 'Max Tile', 
                      os.path.join(output_dir, f'max_tiles_{timestamp}.png'))
    
    if len(agent.loss_history) > 0:
        create_single_plot(agent.loss_history, 'Loss over Training Steps', 'Training Step', 'Loss', 
                          os.path.join(output_dir, f'loss_{timestamp}.png'))
    
    if len(agent.epsilon_history) > 0:
        create_single_plot(agent.epsilon_history, 'Epsilon over Episodes', 'Episode', 'Epsilon', 
                          os.path.join(output_dir, f'epsilon_{timestamp}.png'))
    
    # Save a summary of statistics
    with open(os.path.join(output_dir, f'training_summary_{timestamp}.txt'), 'w') as f:
        f.write(f"Training Statistics:\n")
        f.write(f"Number of Episodes: {len(episode_scores)}\n")
        f.write(f"Final Epsilon: {agent.epsilon:.4f}\n")
        f.write(f"Max Score: {max(episode_scores)}\n")
        f.write(f"Max Tile: {max(episode_max_tiles)}\n")
        f.write(f"Average Score (last 100 episodes): {np.mean(episode_scores[-100:]):.2f}\n")

def create_single_plot(data, title, xlabel, ylabel, filename, window_size=None):
    """Create and save a single plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    
    if window_size and len(data) > window_size:
        avg_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(avg_data) + window_size-1), avg_data, 'r-', label=f'{window_size}-point avg')
        plt.legend()
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_game_state(grid, score, max_tile, action_name=None, output_dir='game_states', filename=None):
    """Visualize and save the current game state as a PNG file"""
    ensure_dir(output_dir)
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'game_state_{timestamp}.png'
    
    plt.figure(figsize=(8, 8))
    
    # Create the grid visualization
    plt.imshow(np.zeros((4, 4, 3)), alpha=0)  # Transparent background
    
    # Define colors for different tile values (based on the original 2048 game)
    tile_colors = {
        0: '#cdc1b4',
        2: '#eee4da',
        4: '#ede0c8',
        8: '#f2b179',
        16: '#f59563',
        32: '#f67c5f',
        64: '#f65e3b',
        128: '#edcf72',
        256: '#edcc61',
        512: '#edc850',
        1024: '#edc53f',
        2048: '#edc22e'
    }
    
    # Define text colors
    text_colors = {
        0: '#776e65',
        2: '#776e65',
        4: '#776e65',
        8: '#f9f6f2',
        16: '#f9f6f2',
        32: '#f9f6f2',
        64: '#f9f6f2',
        128: '#f9f6f2',
        256: '#f9f6f2',
        512: '#f9f6f2',
        1024: '#f9f6f2',
        2048: '#f9f6f2'
    }
    
    # Draw the grid and tiles
    for i in range(4):
        for j in range(4):
            value = grid[i, j]
            color = tile_colors.get(value, '#ff0000')  # Default red for unknown values
            text_color = text_colors.get(value, '#ffffff')
            
            # Draw tile
            rect = plt.Rectangle((j-0.5, i-0.5), 0.9, 0.9, linewidth=2, edgecolor='#bbada0', facecolor=color)
            plt.gca().add_patch(rect)
            
            # Add text
            if value > 0:
                plt.text(j, i, str(value), fontsize=20, ha='center', va='center', color=text_color)
    
    # Add game information
    title = f'Score: {score} | Max Tile: {max_tile}'
    if action_name:
        title += f' | Action: {action_name}'
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return os.path.join(output_dir, filename)

def create_game_gif(image_files, output_filename='game_animation.gif', duration=0.3):
    """Create a GIF from a series of game state images"""
    try:
        from PIL import Image
        
        # Check if there are images to process
        if not image_files:
            print("No images to create GIF")
            return None
        
        # Load all images
        images = [Image.open(f) for f in image_files]
        
        # Save as GIF
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=int(duration * 1000),  # Convert to milliseconds
            loop=0  # 0 means loop forever
        )
        
        return output_filename
    except ImportError:
        print("PIL (Pillow) library is required to create GIFs. Install with: pip install pillow")
        return None
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None 