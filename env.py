
import os
import json
import numpy as np
from PIL import Image

from task import Task

class Node(object):
    """Node"""
    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.dimension = len(resources)
        self.state_matrices = [np.full((duration, resource), 255, dtype=np.uint8) for resource in resources]
        self._state_matrices_capacity = [[resource]*duration for resource in resources]

    def schedule(self, task):
        start_time = self._satisfy(self._state_matrices_capacity, task.resources, task.duration)
        if start_time == -1:
            return False
        else:
            for i in range(0, task.dimension):
                self._occupy(self.state_matrices[i], self._state_matrices_capacity[i], task.resources[i], task.duration, start_time)
            return True

    def timestep(self):
        for i in range(0, self.dimension):
            temp = np.delete(self.state_matrices[i], (0), axis=0)
            temp = np.append(temp, np.array([[255 for x in range(0, temp.shape[1])]]), axis=0)
            self.state_matrices[i] = temp
        for i in range(0, self.dimension):
            self._state_matrices_capacity[i].pop(0)
            self._state_matrices_capacity[i].append(self.resources[i])

    def summary(self, bg_shape=None):
        if self.dimension > 0:
            temp = self._expand(self.state_matrices[0], bg_shape)
            for i in range(1, self.dimension):
                temp = np.concatenate((temp, self._expand(self.state_matrices[i], bg_shape)), axis=1)
            return temp
        else:
            return None

    def plot(self, bg_shape=None):
        if not os.path.exists('__state__'):
            os.makedirs('__state__')
        Image.fromarray(self.summary(bg_shape)).save('__state__/{0}.png'.format(self.label))

    def _satisfy(self, capacity_matrix, required_resources, required_duration):
        p1 = 0
        p2 = 0
        duration_bound = min([len(capacity) for capacity in capacity_matrix])
        while p1 < duration_bound and p2 < required_duration:
            if False in [capacity_matrix[i][p1] >= required_resources[i] for i in range(0, len(required_resources))]:
                p1 = p1 + 1
                p2 = 0
            else:
                p1 = p1 + 1
                p2 = p2 + 1
        if p2 == required_duration:
            return p1 - required_duration
        else:
            return -1

    def _occupy(self, state_matrix, state_matrix_capacity, required_resource, required_duration, start_time):
        for i in range(start_time, start_time+required_duration):
            for j in range(0, required_resource):
                state_matrix[i, len(state_matrix[i])-state_matrix_capacity[i]+j] = 0
            state_matrix_capacity[i] = state_matrix_capacity[i] - required_resource

    def _expand(self, matrix, bg_shape=None):
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

class Environment(object):
    """Environment"""
    def __init__(self, nodes, queue_size, backlog_size):
        self.nodes = nodes
        self.queue_size = queue_size
        self.backlog_size = backlog_size
        self.queue = []
        self.backlog = []

    def summary(self, bg_shape=None):
        if bg_shape is None:
            bg_col = max([max(node.resources) for node in self.nodes])
            bg_row = max([node.duration for node in self.nodes])
            bg_shape = (bg_row, bg_col)
        if len(self.nodes) > 0:
            dimension = self.nodes[0].dimension
            temp = self.nodes[0].summary(bg_shape)
            for i in range(1, len(self.nodes)):
                temp = np.concatenate((temp, self.nodes[i].summary(bg_shape)), axis=1)
            for i in range(0, len(self.queue)):
                temp = np.concatenate((temp, self.queue[i].summary(bg_shape)), axis=1)
            empty_summary = Task([0]*dimension, 0, 'empty_task').summary(bg_shape)
            for i in range(len(self.queue), self.queue_size):
                temp = np.concatenate((temp, empty_summary), axis=1)
            backlog_summary = empty_summary
            p_backlog = 0
            p_row = 0
            p_col = 0
            while p_row < bg_shape[0] and p_col < bg_shape[1] and p_backlog < len(self.backlog):
                backlog_summary[p_row, p_col] = 0
                p_row = p_row + 1
                if p_row == bg_shape[0]:
                    p_row = 0
                    p_col = p_col + 1
                p_backlog = p_backlog + 1
            temp = np.concatenate((temp, backlog_summary), axis=1)
            return temp
        else:
            return None

    def plot(self, bg_shape=None):
        if not os.path.exists('__state__'):
            os.makedirs('__state__')
        Image.fromarray(self.summary(bg_shape)).save('__state__/environment.png')

    def __repr__(self):
        return 'Environment(nodes={0}, queue={1}, backlog={2})'.format(self.nodes, self.queue, self.backlog)

def load_environment():
    """load environment from conf/env.conf.json"""
    with open('conf/env.conf.json', 'r') as fr:
        data = json.load(fr)
        nodes = []
        label= 0
        for node_json in data['nodes']:
            label = label + 1
            nodes.append(Node(node_json['resource_capacity'], node_json['duration_capacity'], 'node' + str(label)))
        return Environment(nodes, data['queue_size'], data['backlog_size'])
