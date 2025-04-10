import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
import os

class ReplayBuffer:
    """Experience replay buffer to store and sample agent experiences"""
    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer"""
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        """Return the current size of the buffer"""
        return len(self.buffer)

def preprocess_state(state):
    """Preprocess the game state for neural network input"""
    # Normalize the state by taking log2 of non-zero values (since tiles are powers of 2)
    # This helps the neural network learn more efficiently
    processed = np.zeros_like(state, dtype=np.float32)
    
    # For each non-zero tile, take log2 and normalize by dividing by 16 (max tile is usually 2^15 = 32768)
    non_zero_mask = state > 0
    processed[non_zero_mask] = np.log2(state[non_zero_mask]) / 16.0
    
    # Reshape to (1, 16) for Dense layers
    return processed.reshape(1, -1)

def create_dqn_model(input_shape, num_actions):
    """Create a deep Q-network model"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

class DQNAgent:
    """Deep Q-Learning agent for playing 2048"""
    
    def __init__(self, state_shape, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, batch_size=64):
        # Environment parameters
        self.state_shape = (state_shape,)  # Input shape for the model (flattened grid)
        self.num_actions = num_actions  # Number of possible actions
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.batch_size = batch_size  # Size of batches to sample from replay buffer
        
        # Create primary and target networks
        self.primary_network = create_dqn_model(self.state_shape, self.num_actions)
        self.target_network = create_dqn_model(self.state_shape, self.num_actions)
        self.update_target_network()
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Metrics for tracking performance
        self.loss_history = []
        self.reward_history = []
        self.max_tile_history = []
        self.epsilon_history = []
    
    def update_target_network(self):
        """Copy weights from primary network to target network"""
        self.target_network.set_weights(self.primary_network.get_weights())
    
    def get_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: choose the best action according to the policy
            q_values = self.primary_network.predict(preprocess_state(state), verbose=0)
            return np.argmax(q_values[0])
    
    def update_replay_buffer(self, state, action, reward, next_state, done):
        """Add a new experience to the replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using experiences from the replay buffer"""
        # Check if we have enough experiences to train
        if self.replay_buffer.size() < self.batch_size:
            return 0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Preprocess states and next_states
        processed_states = np.vstack([preprocess_state(state) for state in states])
        processed_next_states = np.vstack([preprocess_state(next_state) for next_state in next_states])
        
        # Get the current Q values from the primary network
        current_q_values = self.primary_network.predict(processed_states, verbose=0)
        
        # Get the next Q values from the target network
        next_q_values = self.target_network.predict(processed_next_states, verbose=0)
        
        # Initialize the target Q values as the current Q values (we'll update only the chosen actions)
        target_q_values = current_q_values.copy()
        
        # Update the Q values for the actions taken
        for i in range(len(actions)):
            if dones[i]:
                # If the episode ended, there is no next Q value
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # Otherwise, use the Bellman equation to compute the target Q value
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the primary network
        loss = self.primary_network.train_on_batch(processed_states, target_q_values)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save_model(self, filepath):
        """Save the model weights"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.primary_network.save_weights(filepath)
    
    def load_model(self, filepath):
        """Load the model weights"""
        self.primary_network.load_weights(filepath)
        self.update_target_network() 