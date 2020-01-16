
from task import generate_tasks
from task import load_tasks
from env import load_environment

if __name__ == '__main__':
    generate_tasks()
    environment = load_environment()
    print(environment)
    print('done')
