import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
import matplotlib.gridspec as gridspec
from PIL import Image
from utils import ensure_dir
from agent import preprocess_state

def visualize_q_values(agent, state, output_dir='results/decision_viz', filename=None):
    """
    Visualize the Q-values for each action given the current state.
    Shows which action the agent prefers and why.
    """
    ensure_dir(output_dir)
    
    if filename is None:
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'q_values_{timestamp}.png'
    
    # Get state representation ready for network
    processed_state = preprocess_state(state)
    
    # Get Q-values from the network
    q_values = agent.primary_network.predict(processed_state, verbose=0)[0]
    
    # Find the best action
    best_action = np.argmax(q_values)
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2])
    
    # Game grid display
    ax_grid = plt.subplot(gs[:, 0])
    ax_grid.set_title('Current Game State', fontsize=14)
    
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
    
    # Draw the game grid
    ax_grid.imshow(np.zeros((4, 4, 3)), alpha=0)
    for i in range(4):
        for j in range(4):
            value = state[i, j]
            color = tile_colors.get(value, '#ff0000')
            text_color = text_colors.get(value, '#ffffff')
            
            # Draw tile
            rect = Rectangle((j-0.5, i-0.5), 0.9, 0.9, linewidth=2, 
                             edgecolor='#bbada0', facecolor=color)
            ax_grid.add_patch(rect)
            
            # Add text
            if value > 0:
                ax_grid.text(j, i, str(value), fontsize=20, 
                           ha='center', va='center', color=text_color)
    
    # Set limits and hide axes
    ax_grid.set_xlim(-0.6, 3.6)
    ax_grid.set_ylim(3.6, -0.6)
    ax_grid.axis('off')
    
    # Movement visualization with arrows
    ax_moves = plt.subplot(gs[:, 1])
    ax_moves.set_title('Agent\'s Preferred Move', fontsize=14)
    
    # Draw background grid
    ax_moves.imshow(np.zeros((4, 4, 3)), alpha=0)
    for i in range(4):
        for j in range(4):
            rect = Rectangle((j-0.5, i-0.5), 0.9, 0.9, linewidth=1, 
                             edgecolor='#bbada0', facecolor='#cdc1b4', alpha=0.3)
            ax_moves.add_patch(rect)
    
    # Draw large arrow to show best direction
    arrow_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # blue, red, green, orange
    arrow_labels = ['Left', 'Right', 'Up', 'Down']
    
    if best_action == 0:  # left
        arrow = Arrow(3, 1.5, -2, 0, width=1.0, color=arrow_colors[0])
    elif best_action == 1:  # right
        arrow = Arrow(1, 1.5, 2, 0, width=1.0, color=arrow_colors[1])
    elif best_action == 2:  # up
        arrow = Arrow(1.5, 3, 0, -2, width=1.0, color=arrow_colors[2])
    elif best_action == 3:  # down
        arrow = Arrow(1.5, 1, 0, 2, width=1.0, color=arrow_colors[3])
    
    ax_moves.add_patch(arrow)
    ax_moves.text(1.5, -1, f"Chosen Action: {arrow_labels[best_action]}", 
                fontsize=14, ha='center', color=arrow_colors[best_action])
    
    # Set limits and hide axes
    ax_moves.set_xlim(-0.6, 3.6)
    ax_moves.set_ylim(3.6, -0.6)
    ax_moves.axis('off')
    
    # Bar chart for Q-values
    ax_qval = plt.subplot(gs[:, 2])
    ax_qval.set_title('Q-values for Each Action', fontsize=14)
    
    # Create bar chart for Q-values
    bars = ax_qval.bar(range(4), q_values, color=arrow_colors)
    
    # Highlight the best action
    bars[best_action].set_alpha(1.0)
    for i, bar in enumerate(bars):
        if i != best_action:
            bar.set_alpha(0.6)
    
    # Add value labels on top of bars
    for i, v in enumerate(q_values):
        ax_qval.text(i, v + 0.1, f"{v:.2f}", ha='center', fontsize=12)
    
    # Add action labels
    ax_qval.set_xticks(range(4))
    ax_qval.set_xticklabels(arrow_labels)
    ax_qval.set_xlabel('Action', fontsize=12)
    ax_qval.set_ylabel('Q-value', fontsize=12)
    
    # Format y-axis to show relative values better
    min_q = min(q_values) - 0.5
    max_q = max(q_values) + 1
    ax_qval.set_ylim(min_q, max_q)
    
    # Add explanatory text
    ax_qval.text(1.5, min_q + 0.3, "Higher values = Better actions", 
                ha='center', fontsize=12, style='italic')
    
    # Add grid lines for readability
    ax_qval.grid(True, linestyle='--', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return os.path.join(output_dir, filename)

def create_decision_making_animation(agent, env, max_steps=100, output_dir='results/decision_viz'):
    """
    Create a series of visualizations showing the agent's decision-making process
    as it plays through a full game of 2048.
    """
    ensure_dir(output_dir)
    
    # Set agent to evaluation mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # Reset environment to start new game
    state = env.reset()
    done = False
    step = 0
    
    # Save initial state Q-values
    images = []
    img_path = visualize_q_values(
        agent, state, 
        output_dir=output_dir,
        filename=f'step_{step:03d}.png'
    )
    images.append(img_path)
    
    # Play game until done or max steps reached
    while not done and step < max_steps:
        # Get action from agent
        action = agent.get_action(state)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
        step += 1
        
        # Save state and Q-values
        img_path = visualize_q_values(
            agent, state,
            output_dir=output_dir,
            filename=f'step_{step:03d}.png'
        )
        images.append(img_path)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Create GIF animation
    try:
        gif_path = os.path.join(output_dir, 'decision_making_animation.gif')
        frames = [Image.open(img) for img in images]
        
        # Resize all frames to same size
        frames = [img.resize((800, 450)) for img in frames]
        
        # Save as GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,  # 0.5 seconds per frame
            loop=0  # 0 means loop forever
        )
        
        print(f"Created decision-making animation: {gif_path}")
        return gif_path, images
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None, images 