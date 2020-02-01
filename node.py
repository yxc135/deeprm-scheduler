"""Node Manipulation."""

import collections
import os

import numpy as np
from PIL import Image


class Node(object):
    """Node."""

    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(resources)
        self.timestep_counter = 0
        self.scheduled_tasks = []
        # state matrices, 0 stands for occupied slot, 255 stands for vacant slot
        self.state_matrices = [np.full((duration, resource), 255, dtype=np.uint8) for resource in resources]
        self._state_matrices_capacity = [[resource]*duration for resource in resources]

    def schedule(self, task):
        """Schedule task on node."""
        # find the earliest time to schedule task
        start_time = self._satisfy(self._state_matrices_capacity, task.resources, task.duration)

        if start_time == -1:
            # failure, no capacity
            return False
        else:
            # success, update node state
            for i in range(task.dimension):
                self._occupy(self.state_matrices[i], self._state_matrices_capacity[i], task.resources[i], task.duration, start_time)
            self.scheduled_tasks.append((task, self.timestep_counter+task.duration))
            return True

    def timestep(self):
        """Proceed to the next timestep."""
        self.timestep_counter += 1

        # update state matrices
        for i in range(self.dimension):
            temp = np.delete(self.state_matrices[i], (0), axis=0)
            temp = np.append(temp, np.array([[255 for x in range(temp.shape[1])]]), axis=0)
            self.state_matrices[i] = temp
        for i in range(self.dimension):
            self._state_matrices_capacity[i].pop(0)
            self._state_matrices_capacity[i].append(self.resources[i])

        # remove completed tasks
        indices = []
        for i in range(len(self.scheduled_tasks)):
            if self.timestep_counter >= self.scheduled_tasks[i][1]:
                indices.append(i)
        for i in sorted(indices, reverse=True):
            del self.scheduled_tasks[i]

    def summary(self, bg_shape=None):
        """State representation."""
        if self.dimension > 0:
            temp = self._expand(self.state_matrices[0], bg_shape)
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, self._expand(self.state_matrices[i], bg_shape)), axis=1)
            return temp
        else:
            return None

    def utilization(self):
        """Calculate utilization."""
        return sum([collections.Counter(matrix.flatten()).get(0, 0) for matrix in self.state_matrices])/sum(self.resources)/self.duration

    def _satisfy(self, capacity_matrix, required_resources, required_duration):
        """Find the earliest time to schedule task with required resources for required duration."""
        p1 = 0
        p2 = 0
        duration_bound = min([len(capacity) for capacity in capacity_matrix])
        while p1 < duration_bound and p2 < required_duration:
            if False in [capacity_matrix[i][p1] >= required_resources[i] for i in range(len(required_resources))]:
                p1 += 1
                p2 = 0
            else:
                p1 += 1
                p2 += 1
        if p2 == required_duration:
            return p1 - required_duration
        else:
            return -1

    def _occupy(self, state_matrix, state_matrix_capacity, required_resource, required_duration, start_time):
        """Occupy state matrix slots for required resources and duration."""
        for i in range(start_time, start_time+required_duration):
            for j in range(required_resource):
                state_matrix[i, len(state_matrix[i])-state_matrix_capacity[i]+j] = 0
            state_matrix_capacity[i] = state_matrix_capacity[i] - required_resource

    def _expand(self, matrix, bg_shape=None):
        """Expand state matrix to specified background shape."""
        if bg_shape is not None and bg_shape[0] >= matrix.shape[0] and bg_shape[1] >= matrix.shape[1]:
            temp = matrix
            if bg_shape[0] > matrix.shape[0]:
                temp = np.concatenate((temp, np.full((bg_shape[0]-matrix.shape[0], matrix.shape[1]), 255, dtype=np.uint8)), axis=0)
            if bg_shape[1] > matrix.shape[1]:
                temp = np.concatenate((temp, np.full((bg_shape[0], bg_shape[1]-matrix.shape[1]), 255, dtype=np.uint8)), axis=1)
            return temp
        else:
            return matrix

    def __repr__(self):
        return 'Node(state_matrices={0}, label={1})'.format(self.state_matrices, self.label)

