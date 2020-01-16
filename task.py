
import os
import json
import random

class Task(object):
    """Task"""
    def __init__(self, resources, duration, label):
        self.resources = resources
        self.duration = duration
        self.label = label

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
