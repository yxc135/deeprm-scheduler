
from task import generate_tasks
from task import load_tasks
from env import load_environment

from task import Task
from node import Node
from env import Environment
from schedule import CompactScheduler
from schedule import SpreadScheduler

if __name__ == '__main__':
    node1 = Node([100,200,300], 20, 'node1')
    node2 = Node([100,300,300], 20, 'node2')
    node3 = Node([300,300,300], 20, 'node3')
    task = Task([10,20,30], 10, 'sampletask')
    load_environment()
    cs = CompactScheduler()
    ss = SpreadScheduler()
    environment = Environment([node1,node2,node3], 10, 60, cs)
    action = ss.schedule(environment, task)
    while action is not None:
        print(action)
        print([n.utilization() for n in environment.nodes])
        action = ss.schedule(environment, task)
    print('done')
