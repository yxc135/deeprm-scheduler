
import os
import json
import numpy as np
from PIL import Image

from node import Node
from task import Task
from task import generate_tasks
from task import load_tasks
from schedule import CompactScheduler
from schedule import SpreadScheduler
from schedule import DeepRMScheduler

class Environment(object):
    """Environment"""
    def __init__(self, nodes, queue_size, backlog_size, scheduler, task_generator):
        self.nodes = nodes
        self.queue_size = queue_size
        self.backlog_size = backlog_size
        self.queue = []
        self.backlog = []
        self.scheduler = scheduler
        self.task_generator = task_generator
        self.task_generator_end = False
        self.timestep_counter = 0

    def timestep(self):
        self.timestep_counter = self.timestep_counter + 1

        for node in self.nodes:
            node.timestep()

        actions = []
        indices = []
        for i in range(0, len(self.queue)):
            action = self.scheduler.schedule(self, self.queue[i])
            if action is not None:
                actions.append(action)
                indices.append(i)
        for i in sorted(indices, reverse=True):
            del self.queue[i]

        p_queue = len(self.queue)
        p_backlog = 0
        indices = []
        while p_queue < self.queue_size and p_backlog < len(self.backlog):
            self.queue.append(self.backlog[p_backlog])
            indices.append(p_backlog)
            p_queue = p_queue + 1
            p_backlog = p_backlog + 1
        for i in sorted(indices, reverse=True):
            del self.backlog[i]

        p_backlog = len(self.backlog)
        while p_backlog < self.backlog_size:
            new_task = next(self.task_generator, None)
            if new_task is None:
                self.task_generator_end = True
                break
            else:
                self.backlog.append(new_task)
                p_backlog = p_backlog + 1

        return actions

    def terminateMark(self):
        for node in self.nodes:
            if node.utilization() > 0:
                return False
        if self.queue or self.backlog or not self.task_generator_end:
            return False
        return True

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
        summary_matrix = self.summary(bg_shape)
        summary_plot = np.full((summary_matrix.shape[0], summary_matrix.shape[1]), 255, dtype=np.uint8)
        for row in range(0, summary_matrix.shape[0]):
            for col in range(0, summary_matrix.shape[1]):
                summary_plot[row, col] = summary_matrix[row, col]
        Image.fromarray(summary_plot).save('__state__/environment_{0}.png'.format(self.timestep_counter))

    def __repr__(self):
        return 'Environment(timestep_counter={0}, nodes={1}, queue={2}, backlog={3})'.format(self.timestep_counter, self.nodes, self.queue, self.backlog)

def load_environment():
    """load environment from conf/env.conf.json"""
    generate_tasks()
    tasks = load_tasks()
    task_generator = (t for t in tasks)
    with open('conf/env.conf.json', 'r') as fr:
        data = json.load(fr)
        nodes = []
        label= 0
        for node_json in data['nodes']:
            label = label + 1
            nodes.append(Node(node_json['resource_capacity'], node_json['duration_capacity'], 'node' + str(label)))
        if 'CompactScheduler' == data['scheduler']:
            return Environment(nodes, data['queue_size'], data['backlog_size'], CompactScheduler(), task_generator)
        elif 'SpreadScheduler' == data['scheduler']:
            return Environment(nodes, data['queue_size'], data['backlog_size'], SpreadScheduler(), task_generator)
        else:
            return Environment(nodes, data['queue_size'], data['backlog_size'], DeepRMScheduler(), task_generator)
