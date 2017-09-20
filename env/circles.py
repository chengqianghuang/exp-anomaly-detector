import pandas as pd
import numpy as np
import random
import os

from sklearn import datasets

IN = 0
OUT = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [IN, OUT]

def defaultStateFuc(X, index):
    return X[index]
    
def defaultRewardFuc(Y, index, y):
    if y == Y[index]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT

class EnvCircles():
    # init the class instance
    def __init__(self):
        self.action_space_n = len(action_space)
        self.index = 0
        
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc
        
        self.n_samples = 1000
    
    # reset the instance
    def reset(self):
        # 1. select a new time series from the repo and load
        # the time series contains "timestamp", "value", "anomaly"
        self.X, self.Y = datasets.make_circles(n_samples=self.n_samples, factor=.5, noise=.05)
        self.X = self.X/2 + 0.5
        self.index = 0

        # 2. return the first state, containing the first element of the circles
        return self.statefnc(self.X, self.index)
        
    # take a step and gain a reward
    def step(self, action):
        assert(action in action_space)
        
        # 1. get the reward of the action
        reward = self.rewardfnc(self.Y, self.index, action)
        
        # 2. get the next state and the done flag after the action
        self.index += 1
        
        if self.index >= self.n_samples:
            done = 1
            state = []
        else:
            done = 0
            state = self.statefnc(self.X, self.index)
            
        return state, reward, done, []