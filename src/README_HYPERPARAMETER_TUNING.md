# 2048 Agent Hyperparameter Tuning

This module provides tools for systematically testing different hyperparameter configurations for the 2048 reinforcement learning agent. It helps identify the best combination of parameters to maximize agent performance before committing to a full training run.

## Features

- **Rapid Testing**: Test multiple configurations with a smaller number of episodes
- **Predefined Configurations**: Includes common hyperparameter variations to test
- **Comparative Analysis**: Automatically compares performance metrics across configurations
- **Visualization**: Generates learning curves and comparison charts

## Usage

You can run hyperparameter tuning with the following command:

```bash
python -m src.main tune
```

### Options

- `--episodes`: Number of episodes to run per configuration (default: 200)
- `--output-dir`: Directory to save results (default: results/hyperparameter_tuning)
- `--configs`: Comma-separated list of configurations to test, or "all" for all predefined configs (default: all)
- `--log-freq`: Frequency of progress logging in episodes (default: 50)

## Available Configurations

The following predefined configurations are available:

1. **baseline**: Standard configuration with balanced parameters
2. **wider_network**: Network with wider layers (512, 512, 256)
3. **deeper_network**: Network with an additional hidden layer
4. **higher_lr**: Higher learning rate (0.001)
5. **lower_lr**: Lower learning rate (0.00001)
6. **faster_exploration**: Faster epsilon decay for less exploration
7. **slower_exploration**: Slower epsilon decay for more exploration
8. **higher_gamma**: Higher discount factor (0.995)
9. **lower_gamma**: Lower discount factor (0.95)
10. **larger_batch**: Larger batch size (128)
11. **smaller_batch**: Smaller batch size (32)
12. **frequent_target_update**: More frequent target network updates
13. **infrequent_target_update**: Less frequent target network updates

To run specific configurations:

```bash
python -m src.main tune --configs baseline,wider_network,higher_lr
```

## Understanding the Results

After running the tuning process, the results will be saved to the specified output directory with:

1. **Learning Curves**: For each configuration showing score, max tile, loss, and epsilon over episodes
2. **Comparison Charts**: Bar charts comparing key metrics across configurations
3. **CSV Reports**: Detailed metrics for each configuration and a comparison summary

The script will also print a sorted summary of results, ranking configurations by their performance to help you choose the best one.

## What to Look For

When analyzing the results:

- **Higher Average Score**: Better overall performance
- **Higher Max Tile**: Better ability to reach larger tiles
- **Consistent Learning Curves**: Stable and improving scores over time
- **Reasonable Training Time**: Good performance without excessive computation

Once you've identified the best configuration, use those parameters for a full training run:

```bash
python -m src.main train --gamma 0.995 --epsilon-decay 0.9999 --batch-size 128 [other params]
``` 