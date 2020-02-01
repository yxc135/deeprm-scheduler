"""Entrance."""

import env


if __name__ == '__main__':
    environment, scheduler = env.load()
    while not environment.terminated():
        environment.plot()
        actions = scheduler.schedule()
        print(actions)
    print('END')
