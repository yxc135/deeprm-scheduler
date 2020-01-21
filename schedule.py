
from abc import ABC, abstractmethod

class Action(object):
    """Schedule action"""
    def __init__(self, task, node):
        self.task = task
        self.node = node

    def __repr__(self):
        return 'Action(task={0} -> node={1})'.format(self.task.label, self.node.label)

class Scheduler(ABC):
    """Scheduler"""
    @abstractmethod
    def schedule(self, environment):
        pass

class CompactScheduler(Scheduler):
    """Compact scheduler"""
    def schedule(self, environment, task):
        pairs = [(i, environment.nodes[i].utilization()) for i in range(0, len(environment.nodes))]
        pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        for pair in pairs:
            if environment.nodes[pair[0]].schedule(task):
                return Action(task, environment.nodes[pair[0]])
        return None

class SpreadScheduler(Scheduler):
    """Spread scheduler"""
    def schedule(self, environment, task):
        pairs = [(i, environment.nodes[i].utilization()) for i in range(0, len(environment.nodes))]
        pairs = sorted(pairs, key=lambda pair: pair[1])
        for pair in pairs:
            if environment.nodes[pair[0]].schedule(task):
                return Action(task, environment.nodes[pair[0]])
        return None

