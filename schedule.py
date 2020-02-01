
import datetime
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import env

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
    def schedule(self):
        pass

class CompactScheduler(Scheduler):
    """Compact scheduler"""
    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        actions = []
        indices = []
        for i_task in range(0, len(self.environment.queue)):
            pairs = [(i_node, self.environment.nodes[i_node].utilization()) for i_node in range(0, len(self.environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
            for pair in pairs:
                if self.environment.nodes[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.nodes[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]
        self.environment.timestep()
        return actions

class SpreadScheduler(Scheduler):
    """Spread scheduler"""
    def __init__(self, environment):
        self.environment = environment

    def schedule(self):
        actions = []
        indices = []
        for i_task in range(0, len(self.environment.queue)):
            pairs = [(i_node, self.environment.nodes[i_node].utilization()) for i_node in range(0, len(self.environment.nodes))]
            pairs = sorted(pairs, key=lambda pair: pair[1])
            for pair in pairs:
                if self.environment.nodes[pair[0]].schedule(self.environment.queue[i_task]):
                    actions.append(Action(self.environment.queue[i_task], self.environment.nodes[pair[0]]))
                    indices.append(i_task)
                    break
        for i in sorted(indices, reverse=True):
            del self.environment.queue[i]
        self.environment.timestep()
        return actions

class CNNModel(tf.keras.Model):
    """CNN Model"""
    def __init__(self, input_shape, output_shape):
        super(CNNModel, self).__init__()
        if os.path.isfile('__cache__/model/deeprm.h5'):
            self.model = tf.keras.models.load_model('__cache__/model/deeprm.h5')
        else:
            self.model = Sequential([
                Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape),
                MaxPooling2D(),
                Dropout(0.2),
                Flatten(),
                Dense(256, activation='relu'),
                Dense(output_shape, activation='softmax')
            ])

    @tf.function
    def call(self, input_data):
        return self.model(input_data)

    def save(self):
        if not os.path.exists('__cache__/model'):
            os.makedirs('__cache__/model')
        self.model.save('__cache__/model/deeprm.h5')

class DQN(object):
    """DQN Implementation"""
    def __init__(self, input_shape, output_shape):
        self.lr = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.min_experiences = 100
        self.max_experiences = 10000
        self.optimizer = tf.optimizers.Adam(self.lr)
        self.num_actions = output_shape
        self.model = CNNModel(input_shape, output_shape)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def predict(self, input_data):
        return self.model(input_data.astype('float32').reshape(input_data.shape[0], input_data.shape[1], input_data.shape[2], 1))

    @tf.function
    def train(self, dqn_target):
        if len(self.experience['s']) < self.min_experiences:
            return
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        values_next = np.max(dqn_target.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*values_next)
        with tf.GradientTape() as tape:
            predicted_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - predicted_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.array([states]))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, dqn_src):
        variables1 = self.model.trainable_variables
        variables2 = dqn_src.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self):
        self.model.save()

class DeepRMTrainer(object):
    """DeepRM Trainer"""
    def __init__(self, environment):
        self.episodes = 10000
        self.copy_steps = 32
        self.save_steps = 512
        self.epsilon = 0.99
        self.decay = 0.9999
        self.min_epsilon = 0.1
        input_shape = (environment.summary().shape[0], environment.summary().shape[1], 1)
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.dqn_train = DQN(input_shape, output_shape)
        self.dqn_target = DQN(input_shape, output_shape)
        self.total_rewards = np.empty(self.episodes)
        self.environment = environment
        if not os.path.exists('__cache__/summary'):
            os.makedirs('__cache__/summary')
        self.summary_writer = tf.summary.create_file_writer('__cache__/summary/dqn-{0}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train(self):
        for i in range(0, self.episodes):
            self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)
            self.total_rewards[i] = self.train_episode()
            with self.summary_writer.as_default():
                tf.summary.scalar('Episode Reward', self.total_rewards[i], step=i)
            print("Episode {0} Reward {1}".format(i, self.total_rewards[i]))

    def train_episode(self):
        rewards = 0
        step = 0
        self.environment, _ = env.load()
        while not self.environment.terminated():
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, self.epsilon)
            task_index, node_index = self._explain(action_index)
            if task_index < 0 or node_index < 0:
                self.environment.timestep()
                continue
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                self.environment.timestep()
                continue
            del self.environment.queue[task_index]
            prev_observation = observation
            reward = self.environment.reward()
            observation = self.environment.summary()
            rewards = rewards + reward
            exp = {'s': prev_observation, 'a': action_index, 'r': reward, 's2': observation, 'done': self.environment.terminated()}
            self.dqn_train.add_experience(exp)
            self.dqn_train.train(self.dqn_target)
            step = step + 1
            if step % self.copy_steps == 0:
                self.dqn_target.copy_weights(self.dqn_train)
            if step % self.save_steps == 0:
                self.dqn_target.save_weights()
        return rewards

    def _explain(self, action_index):
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit*node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return (task_index, node_index)

class DeepRMScheduler(Scheduler):
    """DeepRM scheduler"""
    def __init__(self, environment):
        input_shape = (environment.summary().shape[0], environment.summary().shape[1], 1)
        output_shape = environment.queue_size * len(environment.nodes) + 1
        self.dqn_train = DQN(input_shape, output_shape)
        self.environment = environment

    def schedule(self):
        actions = []
        while True:
            observation = self.environment.summary()
            action_index = self.dqn_train.get_action(observation, 0)
            task_index, node_index = self._explain(action_index)
            if task_index < 0 or node_index < 0:
               break
            scheduled_task = self.environment.queue[task_index]
            scheduled_node = self.environment.nodes[node_index]
            scheduled = scheduled_node.schedule(scheduled_task)
            if not scheduled:
                break
            del self.environment.queue[task_index]
            actions.append(Action(scheduled_task, scheduled_node))
        self.environment.timestep()
        return actions

    def _explain(self, action_index):
        task_limit = self.environment.queue_size
        node_limit = len(self.environment.nodes)
        if action_index == task_limit*node_limit:
            task_index = -1
            node_index = -1
        else:
            task_index = action_index % task_limit
            node_index = action_index // task_limit
        if task_index >= len(self.environment.queue):
            task_index = -1
            node_index = -1
        return (task_index, node_index)

