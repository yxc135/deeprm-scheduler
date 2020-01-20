
from task import generate_tasks
from task import load_tasks
from env import load_environment

from task import Task
from env import Node
from env import Environment

if __name__ == '__main__':
    node = Node([100,200,300], 20, 'node')
    node.schedule(Task([30,30,30], 10, 'task1'))
    node.plot()
    task = Task([50, 100, 150], 10, 'task')
    task.plot((1000, 1000))
    environment = Environment([node], 10, 60)
    environment.queue.append(task)
    environment.backlog.append(task)
    environment.plot()
    print('done')
