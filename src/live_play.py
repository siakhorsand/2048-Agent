#!/usr/bin/env python3
import os
import sys
import argparse
import time
import pygame
import numpy as np
from game_env import Game2048Env
from agent import DQNAgent, preprocess_state
from decision_viz import visualize_q_values

# Initialize Pygame
pygame.init()

# Colors for the tiles
TILE_COLORS = {
    0: (205, 193, 180),      # Empty tile
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (237, 190, 30),
    8192: (237, 185, 15)
}

# Text colors
TEXT_COLORS = {
    0: (119, 110, 101),
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242),
    4096: (249, 246, 242),
    8192: (249, 246, 242)
}

# Action names
ACTION_NAMES = {
    0: "Left",
    1: "Right",
    2: "Up",
    3: "Down"
}

# Background color
BACKGROUND_COLOR = (187, 173, 160)
GRID_COLOR = (187, 173, 160)
INFO_BG_COLOR = (245, 245, 245)

# Agent thinking color
THINKING_COLOR = (0, 150, 255)  # Blue for thinking
ACTION_COLORS = [
    (52, 152, 219),   # Left (blue)
    (231, 76, 60),    # Right (red)
    (46, 204, 113),   # Up (green)
    (243, 156, 18)    # Down (orange)
]

def render_tile(screen, value, x, y, size):
    """Render a single tile with value at the specified position"""
    # Get colors
    bg_color = TILE_COLORS.get(value, (75, 75, 75))  # Default dark gray for unknown values
    text_color = TEXT_COLORS.get(value, (255, 255, 255))  # Default white for unknown values
    
    # Draw tile background
    pygame.draw.rect(screen, bg_color, (x, y, size, size), border_radius=6)
    
    # Draw value text if the tile is not empty
    if value > 0:
        # Choose font size based on number of digits
        font_size = 48 if value < 100 else 36 if value < 1000 else 24
        font = pygame.font.SysFont("Arial", font_size, bold=True)
        text = font.render(str(value), True, text_color)
        text_rect = text.get_rect(center=(x + size // 2, y + size // 2))
        screen.blit(text, text_rect)

def render_grid(screen, grid, tile_size, spacing):
    """Render the entire grid of the game"""
    grid_size = len(grid)
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = j * (tile_size + spacing) + spacing
            y = i * (tile_size + spacing) + spacing
            render_tile(screen, grid[i][j], x, y, tile_size)

def render_info(screen, info, action, q_values, width, y_start):
    """Render game information and agent decision"""
    font = pygame.font.SysFont("Arial", 24)
    
    # Background for info section
    pygame.draw.rect(screen, INFO_BG_COLOR, (0, y_start, width, 400))
    
    # Game info
    score_text = font.render(f"Score: {info['score']}", True, (0, 0, 0))
    screen.blit(score_text, (20, y_start + 20))
    
    max_tile_text = font.render(f"Max Tile: {info['max_tile']}", True, (0, 0, 0))
    screen.blit(max_tile_text, (20, y_start + 50))
    
    # Agent action
    if action is not None:
        action_text = font.render(f"Agent's Move: {ACTION_NAMES[action]}", True, ACTION_COLORS[action])
        screen.blit(action_text, (20, y_start + 90))
    
    # Render Q-values as a bar chart
    if q_values is not None:
        # Title
        qval_text = font.render("Q-values (higher = better move):", True, (0, 0, 0))
        screen.blit(qval_text, (20, y_start + 130))
        
        # Find min and max for scaling
        max_q = max(q_values)
        min_q = min(q_values)
        range_q = max(max_q - min_q, 1)  # Avoid division by zero
        
        # Bar chart dimensions
        bar_width = 60
        max_bar_height = 120
        bar_spacing = 30
        y_base = y_start + 280
        
        # Draw bars for each action
        for i, q in enumerate(q_values):
            # Normalize q-value to height
            bar_height = int(((q - min_q) / range_q) * max_bar_height)
            
            # Bar position
            x = 40 + i * (bar_width + bar_spacing)
            y = y_base - bar_height
            
            # Draw bar
            pygame.draw.rect(screen, ACTION_COLORS[i], (x, y, bar_width, bar_height))
            
            # Value on top
            value_text = font.render(f"{q:.2f}", True, (0, 0, 0))
            screen.blit(value_text, (x + bar_width//2 - value_text.get_width()//2, y - 25))
            
            # Action label below
            label_text = font.render(ACTION_NAMES[i], True, (0, 0, 0))
            screen.blit(label_text, (x + bar_width//2 - label_text.get_width()//2, y_base + 10))

def live_play(model_path, delay=0.5, save_dir=None, max_steps=200, show_q_values=True):
    """Run the 2048 game with the agent playing in real-time visualization"""
    # Initialize game environment
    env = Game2048Env()
    
    # Initialize agent
    state_shape = 16  # 4x4 grid flattened
    num_actions = 4   # left, right, up, down
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions
    )
    
    # Load the model
    if model_path:
        try:
            agent.load_model(model_path)
            print(f"Loaded model from {model_path}")
            # Set epsilon to 0 for deterministic behavior
            agent.epsilon = 0.0
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Set up save directory if provided
    if save_dir:
        ensure_dir(save_dir)
    
    # Set up Pygame window
    grid_size = 4
    tile_size = 100
    spacing = 10
    screen_width = grid_size * (tile_size + spacing) + spacing
    info_height = 320 if show_q_values else 150
    screen_height = screen_width + info_height
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("2048 AI Agent")
    
    # Game state
    state = env.reset()
    done = False
    steps = 0
    info = {'score': 0, 'max_tile': 0}
    action = None
    q_values = None
    
    # Screenshot counter
    screenshot_count = 0
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    while running and not done and steps < max_steps:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Take a screenshot when space is pressed
                    if save_dir:
                        pygame.image.save(screen, os.path.join(save_dir, f"screenshot_{screenshot_count}.png"))
                        screenshot_count += 1
                        print(f"Screenshot saved: screenshot_{screenshot_count-1}.png")
        
        # Clear screen
        screen.fill(BACKGROUND_COLOR)
        
        # Render current grid
        render_grid(screen, state, tile_size, spacing)
        
        # Render info section
        render_info(screen, info, action, q_values, screen_width, screen_width + 10)
        
        # Update display
        pygame.display.flip()
        
        # Agent thinks and acts
        pygame.display.set_caption("2048 AI Agent (Thinking...)")
        
        # Get Q-values
        if show_q_values:
            processed_state = preprocess_state(state)
            q_values = agent.primary_network.predict(processed_state, verbose=0)[0]
        
        # Choose action
        action = agent.get_action(state)
        
        # Save screenshot before action if requested
        if save_dir and steps % 5 == 0:
            pygame.image.save(screen, os.path.join(save_dir, f"step_{steps}_before.png"))
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
        steps += 1
        
        # Update window title
        pygame.display.set_caption(f"2048 AI Agent | Score: {info['score']} | Max Tile: {info['max_tile']}")
        
        # Save screenshot after action if requested
        if save_dir and steps % 5 == 0:
            pygame.image.save(screen, os.path.join(save_dir, f"step_{steps}_after.png"))
        
        # Wait to make moves visible to human
        time.sleep(delay)
    
    # Game over or stopped
    if done:
        font = pygame.font.SysFont("Arial", 48, bold=True)
        text = font.render("Game Over!", True, (0, 0, 0))
        text_rect = text.get_rect(center=(screen_width // 2, screen_width // 2))
        screen.blit(text, text_rect)
        pygame.display.flip()
        
        # Save final state if requested
        if save_dir:
            pygame.image.save(screen, os.path.join(save_dir, f"game_over_{info['score']}.png"))
        
        print(f"\nGame Over!")
        print(f"Score: {info['score']}")
        print(f"Max Tile: {info['max_tile']}")
        print(f"Steps: {steps}")
        
        # Wait before closing
        time.sleep(3)
    
    pygame.quit()
    return info['score'], info['max_tile'], steps

def ensure_dir(directory):
    """Make sure the directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description='Watch the 2048 RL agent play in real-time')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between moves in seconds')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save screenshots')
    parser.add_argument('--max-steps', type=int, default=200, help='Maximum number of steps')
    parser.add_argument('--hide-q-values', action='store_true', help='Hide Q-values visualization')
    
    args = parser.parse_args()
    
    live_play(
        model_path=args.model_path,
        delay=args.delay,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        show_q_values=not args.hide_q_values
    )

if __name__ == "__main__":
    main() 