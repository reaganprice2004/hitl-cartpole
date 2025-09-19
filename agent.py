"""
agent.py - Minimal DQN agent for CartPole with heavy inline comments.
This file defines a DQNAgent class with methods to select actions, store transitions,
and perform learning steps. It is intentionally explicit and commented line-by-line
to serve as an educational, deployable example for a Human-in-the-Loop project.
"""

# import standard libraries
import random                             # for random action sampling
import numpy as np                        # for numerical arrays and operations
import torch                               # for the neural network and tensors
import torch.nn as nn                      # for neural network layers
import torch.optim as optim                # for optimizer algorithms
from collections import deque             # for an efficient FIFO replay buffer

# import our small ReplayBuffer implementation from the repo
from replay_buffer import ReplayBuffer    # local replay buffer (simple class)

# check device (CPU or CUDA) and set as torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    # initialize the agent with environment dimensions and hyperparameters
    def __init__(self, state_dim, action_dim,
                 hidden_dim=128, lr=1e-3, gamma=0.99,
                 batch_size=64, buffer_size=100000, min_buffer=500):
        # store input / output dimensions
        self.state_dim = state_dim              # dimensionality of state vector
        self.action_dim = action_dim            # number of discrete actions
        # create the online Q-network (a simple MLP)
        self.q_net = nn.Sequential(             # neural network mapping states->Q-values
            nn.Linear(state_dim, hidden_dim),   # fully connected layer
            nn.ReLU(),                          # nonlinearity
            nn.Linear(hidden_dim, hidden_dim),  # another hidden layer
            nn.ReLU(),                          # nonlinearity
            nn.Linear(hidden_dim, action_dim)   # output layer: one Q-value per action
        ).to(device)                            # send the model to the chosen device
        # create the target network (copy of online network for stable targets)
        self.target_net = nn.Sequential(        # same architecture as q_net
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)                            # send to device
        # copy parameters from online q_net to target_net
        self.target_net.load_state_dict(self.q_net.state_dict())
        # create optimizer for q_net parameters
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # discount factor for future rewards
        self.gamma = gamma
        # training minibatch size
        self.batch_size = batch_size
        # replay buffer for experience replay (uses local ReplayBuffer class)
        self.replay_buffer = ReplayBuffer(buffer_size)
        # minimum number of transitions before training begins
        self.min_buffer = min_buffer
        # steps since last target update (used for soft/hard target updates)
        self.steps = 0

    # epsilon-greedy action selection: either explore random or exploit Q-values
    def select_action(self, state, epsilon=0.1):
        # if a random number < epsilon, choose random action (explore)
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        # otherwise, compute Q-values and select argmax (exploit)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # 1xS
        with torch.no_grad():                             # no gradients needed for selection
            q_vals = self.q_net(state_t)                 # forward pass to get Q-values
        return int(q_vals.argmax().cpu().numpy())        # return action index (int)

    # add a transition to replay buffer; transition = (s, a, r, s2, done, is_human)
    def store_transition(self, s, a, r, s2, done, is_human=False):
        # push to buffer; we store a tuple including whether it was a human correction
        self.replay_buffer.add((s, a, r, s2, done, is_human))

    # single gradient descent step using sampled minibatch
    def train_step(self, target_update_freq=1000, human_priority=4.0):
        # if we don't have enough samples, skip training
        if len(self.replay_buffer) < self.min_buffer:
            return None
        # sample a minibatch of transitions
        batch = self.replay_buffer.sample(self.batch_size)
        # unpack batch into arrays
        states = np.stack([b[0] for b in batch]).astype(np.float32)   # (B, state_dim)
        actions = np.array([b[1] for b in batch], dtype=np.int64)     # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)   # (B,)
        next_states = np.stack([b[3] for b in batch]).astype(np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)     # (B,)
        is_human = np.array([1.0 if b[5] else 0.0 for b in batch], dtype=np.float32)
        # convert numpy arrays to torch tensors and send to device
        states_t = torch.tensor(states, dtype=torch.float32).to(device)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        is_human_t = torch.tensor(is_human, dtype=torch.float32).unsqueeze(1).to(device)
        # compute current Q-values for taken actions: Q(s,a)
        q_values = self.q_net(states_t).gather(1, actions_t)  # (B,1)
        # compute target Q-values using target network: r + gamma * max_a' Q_target(s', a') * (1-done)
        with torch.no_grad():
            next_q_vals = self.target_net(next_states_t).max(1)[0].unsqueeze(1)  # (B,1)
            q_targets = rewards_t + (1.0 - dones_t) * (self.gamma * next_q_vals)
        # create weighting term to prioritize human corrections (simple idea: scale loss)
        weights = 1.0 + (human_priority - 1.0) * is_human_t   # human transitions get higher weight
        # compute mean-squared error loss (weighted)
        loss = (weights * (q_values - q_targets).pow(2)).mean()
        # backpropagation step
        self.optimizer.zero_grad()     # reset optimizer gradients
        loss.backward()                # compute gradients
        self.optimizer.step()          # apply parameter update
        # optionally update the target network (hard update every target_update_freq steps)
        self.steps += 1                # increment step counter
        if self.steps % target_update_freq == 0:
            # copy online network parameters into the target network
            self.target_net.load_state_dict(self.q_net.state_dict())
        # return scalar loss for logging
        return float(loss.cpu().detach().numpy())
