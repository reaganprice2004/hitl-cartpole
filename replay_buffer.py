"""
replay_buffer.py - Simple FIFO replay buffer with sampling.
Stores tuples of (s, a, r, s2, done, is_human).
"""
import random                    # for sampling
from collections import deque    # for efficient fixed-size queue

class ReplayBuffer:
    def __init__(self, capacity=100000):
        # create a deque with maxlen to auto-drop old samples when full
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        # append transition to the right (newest)
        self.buffer.append(transition)

    def sample(self, batch_size):
        # sample uniformly at random batch_size transitions
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        # return current number of stored transitions
        return len(self.buffer)
