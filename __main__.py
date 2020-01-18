
from task import generate_tasks
from task import load_tasks
from env import load_environment

from env import Node

if __name__ == '__main__':
    generate_tasks()
    #environment = load_environment()
    #environment.plot()
    node = Node([5,5,5], 3, 'labe')
    node.state_matrices[0][0,0] = 0
    node.state_matrices[1][0,0] = 0
    node.state_matrices[2][0,0] = 0
    print(node)
    node.timestep()
    print(node)
    print('done')
