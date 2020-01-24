
import os
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

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

class DeepRMScheduler(Scheduler):
    """DeepRM scheduler"""
    def __init__(self, environment):
        if not os.path.exists('model'):
            os.makedirs('model')
        if os.path.isfile('model/deeprm.h5'):
            self.model = tf.keras.models.load_model('model/deeprm.h5')
        else:
            input_shape = (environment.summary().shape[0], environment.summary().shape[1], 1)
            output_shape = environment.queue_size * len(environment.nodes) + 1
            self.model = Sequential([
                Conv2D(16, (1, 10), padding='same', activation='relu', input_shape=input_shape),
                MaxPooling2D(),
                Dropout(0.2),
                Conv2D(32, (1, 10), padding='same', activation='relu'),
                MaxPooling2D(),
                Dropout(0.2),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(output_shape, activation='softmax')
            ])
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

    def schedule(self, environment, task):
        pass

