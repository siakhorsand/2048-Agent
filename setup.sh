#!/bin/bash

# Set up the environment for the 2048 RL Agent
echo "Setting up environment for 2048 RL Agent..."

# Detect the OS
OS=$(uname -s)
echo "Detected OS: $OS"

# Check for Python
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "Error: Python not found! Please install Python 3.6 or newer."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create a new requirements file based on the system
TMP_REQ="requirements_tmp.txt"
cat requirements.txt | grep -v "^#" | grep -v "tensorflow" > $TMP_REQ

# Add the appropriate TensorFlow version
if [ "$OS" = "Darwin" ]; then
    # Check if running on Apple Silicon
    if [ "$(uname -m)" = "arm64" ]; then
        echo "Detected Apple Silicon (M1/M2)"
        echo "tensorflow-macos==2.13.0" >> $TMP_REQ
        echo "Using tensorflow-macos for Apple Silicon"
    else
        echo "Detected macOS on Intel"
        echo "tensorflow==2.13.0" >> $TMP_REQ
        echo "Using standard tensorflow"
    fi
else
    echo "Using standard tensorflow"
    echo "tensorflow==2.13.0" >> $TMP_REQ
fi

# Install the dependencies
echo "Installing dependencies..."
$PYTHON -m pip install -r $TMP_REQ

# Clean up
rm $TMP_REQ

# Create PyTorch version of notebook if user has compatibility issues
if [ "$1" = "pytorch" ]; then
    echo "Creating PyTorch version of the notebook..."
    $PYTHON fix_notebook.py
fi

echo "Setup complete! You can now run the Jupyter notebook."
echo "To start Jupyter, run: jupyter notebook 2048_RL_Agent.ipynb" 