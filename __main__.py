
from env import load_environment

if __name__ == '__main__':
    environment = load_environment()
    while not environment.terminateMark():
        environment.timestep()
        environment.plot()
    print('done')
