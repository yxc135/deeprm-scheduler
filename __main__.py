
import env

if __name__ == '__main__':
    environment, scheduler = env.load()
    while not environment.terminated():
        scheduler.schedule()
    print('done')
