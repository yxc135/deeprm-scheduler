
import os
import json
import numpy as np
from PIL import Image

class Node(object):
    """Node"""
    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label
        self.state_matrices = [np.full((duration, resource), 255, dtype=np.uint8) for resource in resources]

    def schedule(self, task):
        return

    def timestep(self):
        for i in range(0, len(self.state_matrices)):
            temp = np.delete(self.state_matrices[i], (0), axis=0)
            temp = np.append(temp, np.array([[255 for x in range(0, temp.shape[1])]]), axis=0)
            self.state_matrices[i] = temp

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

    def plot(self):
        if not os.path.exists('__state__'):
            os.makedirs('__state__')
        for node in self.nodes:
            for i in range(0, len(node.state_matrices)):
                Image.fromarray(node.state_matrices[i]).save('__state__/{0}_resource{1}.png'.format(node.label, i+1))

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
