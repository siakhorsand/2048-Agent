#!/usr/bin/env python3
"""
This script creates a PyTorch-compatible version of the 2048 RL Agent notebook.
"""

import json
import os

def create_pytorch_notebook():
    # Path to the original notebook
    original_notebook_path = '2048_RL_Agent.ipynb'
    # Path for the new PyTorch notebook
    pytorch_notebook_path = '2048_RL_Agent_PyTorch.ipynb'
    
    # Check if the original notebook exists
    if not os.path.exists(original_notebook_path):
        print(f"Error: {original_notebook_path} not found!")
        return
    
    # Read the original notebook
    with open(original_notebook_path, 'r') as f:
        notebook = json.load(f)
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if 'import tensorflow as tf' in ''.join(source):
                new_source = []
                for line in source:
                    if 'import tensorflow' in line:
                        new_source.append('import torch\n')
                        new_source.append('import torch.nn as nn\n')
                        new_source.append('import torch.optim as optim\n')
                    elif 'from tensorflow.keras' in line:
                        continue  # Skip this line
                    elif 'tf.random.set_seed' in line:
                        new_source.append('torch.manual_seed(42)\n')
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                break
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'create_dqn_model' in ''.join(cell['source']):
            notebook['cells'][i]['source'] = [
                "class DQNModel(nn.Module):\n",
                "    \"\"\"PyTorch implementation of the DQN model\"\"\"\n",
                "    \n",
                "    def __init__(self, input_size, output_size):\n",
                "        super(DQNModel, self).__init__()\n",
                "        self.network = nn.Sequential(\n",
                "            nn.Linear(input_size, 256),\n",
                "            nn.ReLU(),\n",
                "            nn.Linear(256, 256),\n",
                "            nn.ReLU(),\n",
                "            nn.Linear(256, 128),\n",
                "            nn.ReLU(),\n",
                "            nn.Linear(128, output_size)\n",
                "        )\n",
                "    \n",
                "    def forward(self, x):\n",
                "        return self.network(x)\n",
                "\n",
                "def create_dqn_model(input_shape, num_actions):\n",
                "    \"\"\"Create a deep Q-network model using PyTorch\"\"\"\n",
                "    return DQNModel(input_shape[0], num_actions)\n"
            ]
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'class DQNAgent' in ''.join(cell['source']):
            # Keep the original implementation for reference but add a comment about PyTorch changes needed
            notebook['cells'][i]['source'].insert(0, "# Note: This needs substantial changes for PyTorch compatibility\n")
            notebook['cells'][i]['source'].insert(1, "# See full implementation in the documentation\n\n")
    
    note_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            "## Important Note About PyTorch Implementation\n\n",
            "This notebook has been partially converted to use PyTorch instead of TensorFlow. The conversion is not complete, and some functions like `DQNAgent.train()` would need to be rewritten for PyTorch.\n\n",
            "Key differences include:\n",
            "- PyTorch uses `torch.Tensor` instead of numpy arrays for neural network inputs\n",
            "- Loss functions and optimizers are handled differently\n",
            "- PyTorch models need to explicitly be put into evaluation mode with `model.eval()` when not training\n",
            "- The training loop would need to be updated to use `loss.backward()` and `optimizer.step()`\n\n",
            "For a full implementation, please refer to PyTorch DQN tutorials online."
        ]
    }
    
    notebook['cells'].insert(1, note_cell)
    
    # Write the  notebook
    with open(pytorch_notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"PyTorch notebook created: {pytorch_notebook_path}")
    print("Note: This is a partial conversion. Some code will still need manual updates.")

if __name__ == "__main__":
    create_pytorch_notebook() 