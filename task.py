
import os
import json
import random

class Task(object):
    """Task"""
    def __init__(self, resources, duration, label):
        self.resource_dimension = len(resources)
        self.resources = resources
        self.duration = duration
        self.label = label

    def __repr__(self):
        return 'Task(resources={0}, duration={1}, label={2})'.format(self.resources, self.duration, self.label)

def load_tasks():
    """load tasks from conf/tasks.csv"""
    tasks = []
    with open('conf/tasks.csv', 'r') as fr:
        resource_indices = []
        duration_index = 0
        label_index = 0
        line = fr.readline()
        parts = line.strip().split(',')
        for i in range(0, len(parts)):
            if parts[i].strip().startswith('resource'):
                resource_indices.append(i)
            if parts[i].strip() == 'duration':
                duration_index = i
            if parts[i].strip() == 'label':
                label_index = i
        line = fr.readline()
        while line:
            parts = line.strip().split(',')
            resources = []
            for index in resource_indices:
                resources.append(int(parts[index]))
            tasks.append(Task(resources, int(parts[duration_index]), parts[label_index]))
            line = fr.readline()
    return tasks

def generate_tasks():
    """generate tasks according to conf/task.pattern.conf.json"""
    if os.path.isfile('conf/tasks.csv'):
        return
    with open('conf/task.pattern.conf.json', 'r') as fr, open('conf/tasks.csv', 'w') as fw:
        data = json.load(fr)
        if len(data) > 0:
            for i in range(0, len(data[0]['resource_range'])):
                fw.write('resource' + str(i+1) + ',')
            fw.write('duration,label' + '\n')
        label = 0
        for task_pattern in data:
            for i in range(0, task_pattern['batch_size']):
                label = label + 1
                resources = []
                duration = str(random.randint(task_pattern['duration_range']['lowerLimit'], task_pattern['duration_range']['upperLimit']))
                for j in range(0, len(task_pattern['resource_range'])):
                    resources.append(str(random.randint(task_pattern['resource_range'][j]['lowerLimit'], task_pattern['resource_range'][j]['upperLimit'])))
                fw.write(','.join(resources) + ',' + duration +  ',' + 'task' + str(label) + '\n')
