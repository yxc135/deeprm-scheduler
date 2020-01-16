
import json

class Node(object):
    """Node"""
    def __init__(self, resources, duration, label):
        self.resource_dimension = len(resources)
        self.resources = resources
        self.duration = duration
        self.label = label

    def __repr__(self):
        return 'Node(resources={0}, duration={1}, label={2})'.format(self.resources, self.duration, self.label)

class Environment(object):
    """Environment"""
    def __init__(self, nodes, queue_size, backlog_size):
        self.nodes = nodes
        self.queue_size = queue_size
        self.backlog_size = backlog_size
        self.queue = []
        self.backlog = []

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
