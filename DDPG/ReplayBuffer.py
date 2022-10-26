import random
from collections import deque
import numpy as np

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, next_state, action, reward, done):
        experience = (state, next_state, action, np.array([reward]), done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, next_state, action, reward, done = experience
            state_batch.append(state)
            next_state_batch.append(next_state)
            action_batch.append(action)
            reward_batch.append(reward)
            done_batch.append(done)
        
        return state_batch, next_state_batch, action_batch, reward_batch, done_batch

    def __len__(self):
        return len(self.buffer)