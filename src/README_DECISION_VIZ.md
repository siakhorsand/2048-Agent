# 2048 Agent Decision Visualization

This module visualizes the decision-making process of the reinforcement learning agent while playing the 2048 game. It helps understand how the agent evaluates different actions and why it makes specific choices.

## Features

- **Q-value Visualization**: Shows the Q-values for each possible action (left, right, up, down) as a bar chart.
- **Current Game State**: Displays the current 2048 game board.
- **Best Action Visualization**: Shows the preferred action with an arrow.
- **Animation**: Creates GIF animations of the agent's decision-making process over a full game.

## Usage

You can use the decision visualization with the following command:

```bash
python -m src.main visualize-decisions --model-path models/trained_model.h5
```

### Options

- `--model-path`: Path to the trained model weights (required)
- `--output-dir`: Directory to save visualizations (default: results/decision_viz)
- `--max-steps`: Maximum number of steps to run (default: 100)
- `--visualize-only`: Only create a single visualization for the initial state without making a GIF

## Visualization Explained

The visualization consists of three parts:

1. **Game State Panel (left)**: Shows the current 2048 board.

2. **Agent's Decision Panel (middle)**: Shows which direction the agent prefers to move.
   - The arrow indicates the chosen direction (left, right, up, or down).
   - This is determined by finding the action with the highest Q-value.

3. **Q-values Panel (right)**: Displays a bar chart of the Q-values for each possible action.
   - Higher values indicate actions that the agent expects to lead to better outcomes.
   - These are the raw outputs from the agent's neural network.
   - The highest bar corresponds to the action the agent will take.

## Understanding Q-values

In reinforcement learning, Q-values represent the expected future reward for taking a specific action in a given state. The agent learns these Q-values through training and uses them to make decisions:

- **Higher Q-value** = Better expected outcome
- Agent always chooses the action with the highest Q-value during evaluation
- Q-values change based on the game state

## Example Output

The tool will create:

1. Individual PNG files for each step of the game showing the decision-making process
2. A GIF animation combining all steps (unless `--visualize-only` is specified)

The outputs will be saved in the specified `--output-dir` directory. 