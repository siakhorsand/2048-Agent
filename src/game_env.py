import numpy as np
import random

class Game2048Env:
    """2048 game environment adapted for reinforcement learning"""
    
    # Action mappings
    ACTIONS = {
        0: 'left',
        1: 'right',
        2: 'up',
        3: 'down'
    }
    
    def __init__(self):
        """Initialize the game environment"""
        self.size = 4
        self.grid = None
        self.score = 0
        self.done = False
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.done = False
        
        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()
        
        return self._get_state()
    
    def _add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        if not self._has_empty_cells():
            return False
        
        # Find empty cells
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 0]
        
        # Choose a random empty cell
        i, j = random.choice(empty_cells)
        
        # Add a 2 (90% chance) or 4 (10% chance)
        self.grid[i, j] = 2 if random.random() < 0.9 else 4
        
        return True
    
    def _has_empty_cells(self):
        """Check if there are any empty cells"""
        return 0 in self.grid
    
    def _compress_row(self, row):
        """Move all non-zero elements to the left"""
        # Remove zeros
        new_row = np.array([x for x in row if x != 0])
        # Add zeros at the end
        new_row = np.append(new_row, np.zeros(self.size - len(new_row), dtype=int))
        return new_row
    
    def _merge_row(self, row):
        """Merge tiles of the same value in a row"""
        score_increase = 0
        
        # Iterate through the list from left to right
        i = 0
        while i < self.size - 1:
            # If current and next element are the same and not zero
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                score_increase += row[i]
                row[i + 1] = 0
                i += 2
            else:
                i += 1
                
        return row, score_increase
    
    def _move_left(self):
        """Move all tiles to the left and merge if possible"""
        score_increase = 0
        changed = False
        
        for i in range(self.size):
            original_row = self.grid[i, :].copy()
            
            # Compress (move non-zero elements to the left)
            row = self._compress_row(original_row)
            
            # Merge
            row, score = self._merge_row(row)
            score_increase += score
            
            # Compress again after merging
            row = self._compress_row(row)
            
            # Update grid
            if not np.array_equal(original_row, row):
                changed = True
                self.grid[i, :] = row
        
        return changed, score_increase
    
    def _move_right(self):
        """Move all tiles to the right and merge if possible"""
        score_increase = 0
        changed = False
        
        for i in range(self.size):
            original_row = self.grid[i, :].copy()
            
            # Reverse the row
            row = original_row[::-1]
            
            # Compress (move non-zero elements to the left)
            row = self._compress_row(row)
            
            # Merge
            row, score = self._merge_row(row)
            score_increase += score
            
            # Compress again after merging
            row = self._compress_row(row)
            
            # Reverse back
            row = row[::-1]
            
            # Update grid
            if not np.array_equal(original_row, row):
                changed = True
                self.grid[i, :] = row
        
        return changed, score_increase
    
    def _move_up(self):
        """Move all tiles up and merge if possible"""
        score_increase = 0
        changed = False
        
        # Transpose the grid
        self.grid = self.grid.T
        
        # Apply left move logic to each row (which are now columns)
        changed, score_increase = self._move_left()
        
        # Transpose back
        self.grid = self.grid.T
        
        return changed, score_increase
    
    def _move_down(self):
        """Move all tiles down and merge if possible"""
        score_increase = 0
        changed = False
        
        # Transpose the grid
        self.grid = self.grid.T
        
        # Apply right move logic to each row (which are now columns)
        changed, score_increase = self._move_right()
        
        # Transpose back
        self.grid = self.grid.T
        
        return changed, score_increase
    
    def _get_max_tile(self):
        """Get the value of the highest tile on the board"""
        return np.max(self.grid)
    
    def _get_state(self):
        """Return the current state of the game"""
        return self.grid.copy()
    
    def _is_game_over(self):
        """Check if the game is over (no empty cells and no possible merges)"""
        # Check for empty cells
        if self._has_empty_cells():
            return False
        
        # Check for possible merges horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.grid[i, j] == self.grid[i, j + 1]:
                    return False
        
        # Check for possible merges vertically
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.grid[i, j] == self.grid[i + 1, j]:
                    return False
        
        return True
    
    def step(self, action):
        """Take an action (0: left, 1: right, 2: up, 3: down) and return new state, reward, done"""
        if self.done:
            # If game is already over, return current state with zero reward
            return self._get_state(), 0, True, {}
        
        prev_score = self.score
        prev_max_tile = self._get_max_tile()
        moved = False
        
        # Execute the move
        if action == 0:  # left
            moved, score_increase = self._move_left()
        elif action == 1:  # right
            moved, score_increase = self._move_right()
        elif action == 2:  # up
            moved, score_increase = self._move_up()
        elif action == 3:  # down
            moved, score_increase = self._move_down()
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Update score if the move was valid
        if moved:
            self.score += score_increase
            self._add_random_tile()
        
        # Check if game is over
        self.done = self._is_game_over()
        
        # Calculate reward
        current_max_tile = self._get_max_tile()
        
        # Three components to the reward:
        # 1. Points gained from merging tiles
        # 2. Bonus for creating a new highest tile
        # 3. Penalty for invalid moves
        
        # Base reward is the score increase
        reward = score_increase
        
        # Bonus for new max tile
        if current_max_tile > prev_max_tile:
            reward += current_max_tile  # Bonus equal to the value of the new max tile
        
        # Penalty for invalid moves
        if not moved:
            reward -= 10  # Small penalty for trying an invalid move
        
        # Penalty for losing the game
        if self.done:
            reward -= 50  # Larger penalty for ending the game
        
        # Return state, reward, done, and info dictionary
        info = {
            'score': self.score,
            'max_tile': current_max_tile,
            'moved': moved
        }
        
        return self._get_state(), reward, self.done, info
    
    def render(self, mode='human'):
        """Display the current state of the game"""
        if mode != 'human':
            return
            
        # Convert 0s to empty strings for cleaner display
        display_grid = np.where(self.grid > 0, self.grid, '')
        
        # Print the grid
        print(f"Score: {self.score}")
        print("+------+------+------+------+")
        for i in range(self.size):
            row_str = "|"                    
            for j in range(self.size):
                cell = display_grid[i, j]
                if cell == '':
                    row_str += "      |"
                else:
                    row_str += f" {cell:4d} |"
            print(row_str)
            print("+------+------+------+------+")
        
        if self.done:
            print("Game Over!") 