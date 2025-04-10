import pygame
import random
import sys
import os

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 450  # Increased height to accommodate score display
GRID_SIZE = 4
TILE_SIZE = SCREEN_WIDTH // GRID_SIZE
PADDING = 10
HIGH_SCORE_FILE = "high_score.txt"

# Colors (based on the original 2048 game)
BACKGROUND_COLOR = (187, 173, 160)
EMPTY_TILE_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
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
    2048: (237, 194, 46)
}

TEXT_COLORS = {
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
    2048: (249, 246, 242)
}

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('2048')

# Initialize fonts
pygame.font.init()
font = pygame.font.SysFont(None, 48)
score_font = pygame.font.SysFont(None, 36)

def initialize_grid():
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    place_random_tile(grid)
    place_random_tile(grid)
    return grid

def place_random_tile(grid):
    empty_positions = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c] == 0]
    if empty_positions:
        r, c = random.choice(empty_positions)
        grid[r][c] = random.choice([2, 4])
        return True
    return False

def compress_row(row):
    # Remove zeros
    new_row = [x for x in row if x != 0]
    # Add zeros at the end
    new_row += [0] * (GRID_SIZE - len(new_row))
    return new_row

def merge_row(row):
    score_increase = 0
    
    # Iterate through the list from left to right
    i = 0
    while i < GRID_SIZE - 1:
        # If current and next element are the same and not zero
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            score_increase += row[i]
            row[i + 1] = 0
            i += 2
        else:
            i += 1
            
    return row, score_increase

def transpose_grid(grid):
    transposed = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            transposed[i][j] = grid[j][i]
    return transposed

def move_left(grid):
    total_score = 0
    changed = False
    
    for i in range(GRID_SIZE):
        original_row = grid[i][:]
        
        row = compress_row(original_row)
        row, score = merge_row(row)
        total_score += score
        row = compress_row(row)
        
        if original_row != row:
            changed = True
            grid[i] = row
    
    return grid, changed, total_score

def move_right(grid):
    total_score = 0
    changed = False
    
    for i in range(GRID_SIZE):
        original_row = grid[i][:]
        
        row = original_row[::-1]
        row = compress_row(row)
        row, score = merge_row(row)
        total_score += score
        row = compress_row(row)
        row = row[::-1]
        
        if original_row != row:
            changed = True
            grid[i] = row
    
    return grid, changed, total_score

def move_up(grid):
    total_score = 0
    changed = False
    
    grid = transpose_grid(grid)
    
    for i in range(GRID_SIZE):
        original_row = grid[i][:]
        
        row = compress_row(original_row)
        row, score = merge_row(row)
        total_score += score
        row = compress_row(row)
        
        if original_row != row:
            changed = True
            grid[i] = row
    
    grid = transpose_grid(grid)
    
    return grid, changed, total_score

def move_down(grid):
    total_score = 0
    changed = False
    
    grid = transpose_grid(grid)
    
    for i in range(GRID_SIZE):
        original_row = grid[i][:]
        
        row = original_row[::-1]
        row = compress_row(row)
        row, score = merge_row(row)
        total_score += score
        row = compress_row(row)
        row = row[::-1]
        
        if original_row != row:
            changed = True
            grid[i] = row
    
    grid = transpose_grid(grid)
    
    return grid, changed, total_score

def check_game_over(grid):
    # Check for empty cells
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 0:
                return False
    
    # Check for possible merges horizontally
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if grid[i][j] == grid[i][j + 1]:
                return False
    
    # Check for possible merges vertically
    for i in range(GRID_SIZE - 1):
        for j in range(GRID_SIZE):
            if grid[i][j] == grid[i + 1][j]:
                return False
    
    return True

def check_win(grid):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 2048:
                return True
    return False

def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            f.write(str(score))
    except:
        # If the file can't be written, just continue without saving
        pass

def load_high_score():
    try:
        if os.path.exists(HIGH_SCORE_FILE):
            with open(HIGH_SCORE_FILE, 'r') as f:
                return int(f.read())
        return 0
    except:
        # If the file can't be read or is corrupt, return 0
        return 0

def draw_grid(grid):
    screen.fill(BACKGROUND_COLOR)
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Calculate tile position with padding
            x = col * TILE_SIZE + PADDING
            y = row * TILE_SIZE + PADDING
            width = TILE_SIZE - 2 * PADDING
            height = TILE_SIZE - 2 * PADDING
            
            value = grid[row][col]
            color = TILE_COLORS.get(value, (237, 194, 46))
            
            pygame.draw.rect(screen, color, (x, y, width, height), border_radius=5)
            
            if value != 0:
                text_color = TEXT_COLORS.get(value, (255, 255, 255))
                text = font.render(str(value), True, text_color)
                text_rect = text.get_rect(center=(x + width // 2, y + height // 2))
                screen.blit(text, text_rect)

def draw_scores(score, high_score):
    # Draw score background
    pygame.draw.rect(screen, (238, 228, 218), (10, 400, SCREEN_WIDTH / 2 - 15, 40), border_radius=5)
    pygame.draw.rect(screen, (238, 228, 218), (SCREEN_WIDTH / 2 + 5, 400, SCREEN_WIDTH / 2 - 15, 40), border_radius=5)
    
    score_text = score_font.render(f"Score: {score}", True, (119, 110, 101))
    high_score_text = score_font.render(f"Best: {high_score}", True, (119, 110, 101))
    
    screen.blit(score_text, (20, 410))
    screen.blit(high_score_text, (SCREEN_WIDTH / 2 + 15, 410))

def draw_game_over():
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((238, 228, 218, 150))
    screen.blit(overlay, (0, 0))
    
    font_go = pygame.font.SysFont(None, 64)
    text_go = font_go.render("Game Over!", True, (119, 110, 101))
    text_rect_go = text_go.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
    
    font_restart = pygame.font.SysFont(None, 36)
    text_restart = font_restart.render("Press 'R' to Restart", True, (119, 110, 101))
    text_rect_restart = text_restart.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
    
    screen.blit(text_go, text_rect_go)
    screen.blit(text_restart, text_rect_restart)

def draw_win():
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((237, 194, 46, 150))
    screen.blit(overlay, (0, 0))
    
    font_win = pygame.font.SysFont(None, 64)
    text_win = font_win.render("You Win!", True, (249, 246, 242))
    text_rect_win = text_win.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
    
    font_continue = pygame.font.SysFont(None, 36)
    text_continue = font_continue.render("Press 'C' to Continue", True, (249, 246, 242))
    text_rect_continue = text_continue.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
    
    screen.blit(text_win, text_rect_win)
    screen.blit(text_continue, text_rect_continue)

def display_controls():
    instructions = [
        "Controls:",
        "Arrow Keys or WASD - Move tiles",
        "R - Restart game",
        "C - Continue after winning"
    ]
    
    y_position = 10
    for instruction in instructions:
        text = score_font.render(instruction, True, (238, 228, 218))
        text_rect = text.get_rect(topleft=(10, y_position))
        screen.blit(text, text_rect)
        y_position += 25

def main():
    grid = initialize_grid()
    game_over = False
    win = False
    win_displayed = False
    score = 0
    high_score = load_high_score()
    clock = pygame.time.Clock()
    show_controls = True
    controls_timer = 180  # Show controls for about 3 seconds (60 frames per second)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save high score before quitting
                if score > high_score:
                    save_high_score(score)
                pygame.quit()
                sys.exit()
            
            if not game_over:
                if event.type == pygame.KEYDOWN:
                    changed = False
                    score_increase = 0
                    
                    # Show controls when any key is pressed
                    if show_controls:
                        show_controls = False
                    
                    # Arrow keys and WASD support
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        grid, changed, score_increase = move_left(grid)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        grid, changed, score_increase = move_right(grid)
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        grid, changed, score_increase = move_up(grid)
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        grid, changed, score_increase = move_down(grid)
                    
                    if changed:
                        score += score_increase
                        
                        # Update high score if needed
                        if score > high_score:
                            high_score = score
                            save_high_score(high_score)
                            
                        place_random_tile(grid)
                        
                        # Check for win condition
                        if not win and check_win(grid):
                            win = True
                    
                    # If win is displayed and user presses 'C', continue playing
                    if win_displayed and event.key == pygame.K_c:
                        win_displayed = False
            else:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    grid = initialize_grid()
                    game_over = False
                    win = False
                    win_displayed = False
                    score = 0
                    # Note: high_score is preserved
        
        draw_grid(grid)
        draw_scores(score, high_score)
        
        # Display controls at the start or for a brief period
        if show_controls:
            display_controls()
            controls_timer -= 1
            if controls_timer <= 0:
                show_controls = False
        
        if not game_over and not win_displayed:
            game_over = check_game_over(grid)
        
        if win and not win_displayed:
            draw_win()
            win_displayed = True
        
        if game_over:
            draw_game_over()
        
        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main() 