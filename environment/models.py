# -*- coding: utf-8 -*-
import random
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.models import load_model
import h5py
import time
import game
from pystockfish import *
from util import *

########### TEACHING AGENT #############################

class TeacherAgent:
    def __init__(self, state_size=4, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.8142083663757826
        self.epsilon = 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.002698726297401723
        self.model = self._build_model()
        self.moves_since_hint = 0
        self.not_yet_rewarded = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.moves_since_hint == 2: #Meaning the teacher waits at most 3 moves for a reward, could be tweaked
            self.moves_since_hint = 0
            return 2 #This enforces that we don't go too long without giving hints (could change to full OR partial hint l8r)
        if np.random.rand() <= self.epsilon:
            random_index = random.randrange(self.action_size)
            if random_index == 0:
                self.moves_since_hint += 1
            else:
                self.moves_since_hint = 0
            return random_index
        act_values = self.model.predict(state)
        nonrandom_index = np.argmax(act_values[0])
        if nonrandom_index == 0:
            self.moves_since_hint += 1
        else:
            self.moves_since_hint = 0
        return nonrandom_index

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, int(batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # if not done: #USED TO BE COMMENTED IN
            #print ("teacher agent state (should be an array with shape (4, )): ", next_state)
            # target = (reward + self.gamma *
            #           np.amax(self.model.predict(next_state)[0]))
            whole_list = self.model.predict(next_state)
            amax_result = np.amax(whole_list[0])
            target = reward + self.gamma * amax_result
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

###############################################################################
# DQN Model
###############################################################################

class StudentAgent:
    def __init__(self, state_size=18, action_size=1856):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9213798872899134
        self.epsilon = 0.5
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0042981037511488785
        self.model = self._build_model()
        self.batch_size = 8

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_move_indices):
        if np.random.rand() <= self.epsilon:
            random_action_index = random.choice(valid_move_indices)
            return random_action_index
        act_values = self.model.predict(state)
        new_act_values = []
        for i,val in enumerate(act_values[0]):
            if i in valid_move_indices:
                new_act_values.append(val)
            else:
                new_act_values.append(0.0)
        startIndex = random.randint(0, len(new_act_values) - 1)
        newVals = []
        for i in range(len(new_act_values) - startIndex):
            newVals.append(new_act_values[startIndex + i])
        for j in range(startIndex):
            newVals.append(new_act_values[j])
        rotated_index = np.argmax(newVals)
        return (rotated_index + startIndex) % len(new_act_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, int(batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # if not done: USED TO BE NOT COMMENTED OUT
                #print ("student agent state length (should be 18): ", len(next_state))
            target = (reward + self.gamma *
                      np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
