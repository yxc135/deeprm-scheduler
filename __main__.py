
from task import generate_tasks
from task import load_tasks
from env import load_environment

from task import Task
from env import Node

if __name__ == '__main__':
    node = Node([5,5,5], 5, 'node')
    node.schedule(Task([3,3,3], 2, 'task1'))
    node.plot([500,500])
    task = Task([50, 100, 150], 50, 'task')
    task.plot((1000, 1000))
    print('done')
