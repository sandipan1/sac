import numpy as np
import random
class ReplayBuffer():
    def __init__(self,size):
        self.size = size
        self.buffer = []
        self.position = 0
    def add(self, state, action, reward, next_state, done):
        """
        :params: transition at each time step

        """
        if len(self.buffer) > self.size :
            self.buffer.append(None)
        self.buffer[self.position] = [state, action, reward , next_state, done]

        self.position = (self.position +1) % self.size
