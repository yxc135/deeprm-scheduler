
import os
import numpy as np
from PIL import Image

class Task(object):
    """Task"""
    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(resources)

    def summary(self, bg_shape=None):
        if bg_shape is None:
            bg_shape = (self.duration, max(self.resources))
        if self.dimension > 0:
            state_matrices = [np.full(bg_shape, 255, dtype=np.uint8) for i in range(0, self.dimension)]
            for i in range(0, self.dimension):
                for row in range(0, self.duration):
                    for col in range(0, self.resources[i]):
                        state_matrices[i][row, col] = 0
            temp = state_matrices[0]
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, state_matrices[i]), axis=1)
            return temp
        else:
            return None

    def __repr__(self):
        return 'Task(resources={0}, duration={1}, label={2})'.format(self.resources, self.duration, self.label)

