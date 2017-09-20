import pandas as pd
import numpy as np
import random
import os

import sklearn.preprocessing

NOT_ANOMALY = 0
ANOMALY = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]

# get all the path of the csv files to be loaded 
#repodir = '../env/time_series_repo/'
repodir = 'env/time_series_repo/'
repodirext = []

for subdir, dirs, files in os.walk(repodir):
    for file in files:
        if file.find('.csv') != -1:
            repodirext.append(os.path.join(subdir, file))

# each csv file is read as the following, therefore contains three columns:
# - timestamp
# - value
# - anomaly
# pd.read_csv(repodirext[random.randint(0, len(repodirext)-1)], usecols=[0,1,2], \
#                       header=0, names=['timestamp','value','anomaly'])

def defaultStateFuc(timeseries, timeseries_curser):
    return timeseries['value'][timeseries_curser]
    
def defaultRewardFuc(timeseries, timeseries_curser, action):
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT

class EnvTimeSeriesfromRepo():
    # init the class instance
    def __init__(self):
        self.action_space_n = len(action_space)
        
        self.timeseries = []
        self.timeseries_curser = -1
        
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc
        
        self.datasetfix = 0
        self.datasetidx = random.randint(0, len(repodirext)-1)
        
        self.datasetsize = len(repodirext)
    
    # reset the instance
    def reset(self):
        # 1. select a new time series from the repo and load
        # the time series contains "timestamp", "value", "anomaly"
        if self.datasetfix == 0:        
	        self.datasetidx = random.randint(0, len(repodirext)-1)
	        
        self.timeseries = pd.read_csv(repodirext[self.datasetidx], \
                                    usecols=[0,1,2], header=0, names=['timestamp','value','anomaly'])
        self.timeseries_curser = 0

        # 2. Preprocess the time series values
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(np.array(self.timeseries['value']).reshape(-1,1))
        self.timeseries['value'] = scaler.transform(np.array(self.timeseries['value']).reshape(-1,1))
        
        # 3. return the first state, containing the first element of the time series
        state = self.statefnc(self.timeseries, self.timeseries_curser)
        
        return state
        
    # return the whole dataset
    def reset_getall(self):
        # 1. select a new time series from the repo and load
        # the time series contains "timestamp", "value", "anomaly"
        if self.datasetfix == 0:        
	        self.datasetidx = random.randint(0, len(repodirext)-1)
	        
        self.timeseries = pd.read_csv(repodirext[self.datasetidx], \
                                    usecols=[0,1,2], header=0, names=['timestamp','value','anomaly'])
        self.timeseries_curser = 0

        # 2. Preprocess the time series values
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(np.array(self.timeseries['value']).reshape(-1,1))
        self.timeseries['value'] = scaler.transform(np.array(self.timeseries['value']).reshape(-1,1))
        
        return self.timeseries
        
    # take a step and gain a reward
    def step(self, action):
        assert(action in action_space)
        assert(self.timeseries_curser >= 0)
        
        # 1. get the reward of the action
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        
        # 2. get the next state and the done flag after the action
        self.timeseries_curser += 1
        
        if self.timeseries_curser >= self.timeseries['value'].size:
            done = 1
            state = []
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser)
            
        return state, reward, done, []