
import os
import json
import numpy as np
from PIL import Image

from node import Node
from task import Task

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
