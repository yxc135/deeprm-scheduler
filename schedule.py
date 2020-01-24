
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

class Action(object):
    """Schedule action"""
    def __init__(self, task_index, node_index, task, node):
        self.task_index = task_index
        self.node_index = node_index
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
    def schedule(self, environment):
        actions = []
        for i_task in range(0, len(environment.queue)):
            pairs = [(i_node, environment.nodes[i_node].utilization()) for i_node in range(0, len(environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
            for pair in pairs:
                if environment.nodes[pair[0]].schedule(environment.queue[i_task]):
                    actions.append(Action(i_task, pair[0], environment.queue[i_task], environment.nodes[pair[0]]))
                    break
        return actions

class SpreadScheduler(Scheduler):
    """Spread scheduler"""
    def schedule(self, environment):
        actions = []
        for i_task in range(0, len(environment.queue)):
            pairs = [(i_node, environment.nodes[i_node].utilization()) for i_node in range(0, len(environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1])
            for pair in pairs:
                if environment.nodes[pair[0]].schedule(environment.queue[i_task]):
                    actions.append(Action(i_task, pair[0], environment.queue[i_task], environment.nodes[pair[0]]))
                    break
        return actions

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

    def schedule(self, environment):
        summary = environment.summary()
        probs = self.model.predict(np.array([summary.reshape(summary.shape[0], summary.shape[1], 1)]))[0]
        action_probs = []
        n_task = environment.queue_size
        n_node = len(environment.nodes)
        for i in range(0, len(probs)):
            if i == n_task*n_node:
                action_probs.append((None, probs[i]))
            else:
                i_task = i % n_task
                i_node = i // n_task
                action_probs.append((Action(i_task, i_node, None, None), probs[i]))
        action_probs = sorted(action_probs, key=lambda action_prob: action_prob[1], reverse=True)

        actions = []
        scheduled_tasks = set()
        for action_prob in action_probs:
            if action_prob[0] is None:
                break
            if action_prob[0].task_index >= len(environment.queue) or action_prob[0].task_index in scheduled_tasks:
                continue
            action = Action(
                action_prob[0].task_index, action_prob[0].node_index,
                environment.queue[action_prob[0].task_index], environment.nodes[action_prob[0].node_index]
            )
            if action.node.schedule(action.task):
                actions.append(action)
                scheduled_tasks.add(action.task_index)
        return actions

