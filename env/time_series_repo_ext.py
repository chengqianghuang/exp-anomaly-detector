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
    	
# each csv file in Yahoo Benchmark is read as the following, 
# therefore contains two columns:
# - value
# - anomaly
# pd.read_csv(self.repodirext[random.randint(0, len(self.repodirext)-1)], usecols=[1,2], \
#                       header=0, names=['value','anomaly'])

def defaultStateFuc(timeseries, timeseries_curser):
    return timeseries['value'][timeseries_curser]
    
def defaultRewardFuc(timeseries, timeseries_curser, action):
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT

class EnvTimeSeriesfromRepo():
    # init the class instance
    def __init__(self, repodir='env/time_series_repo/'):
    
		# get all the path of the csv files to be loaded
		self.repodir = repodir
		self.repodirext = []
		
		for subdir, dirs, files in os.walk(self.repodir):
			for file in files:
				if file.find('.csv') != -1:
					self.repodirext.append(os.path.join(subdir, file))
		
		self.action_space_n = len(action_space)
		
		self.timeseries = []
		self.timeseries_curser = -1
		self.timeseries_curser_init = 0
		self.timeseries_states = []
        
		self.statefnc = defaultStateFuc
		self.rewardfnc = defaultRewardFuc
        
		self.datasetsize = len(self.repodirext)
        
		self.datasetfix = 0
		self.datasetidx = random.randint(0, len(self.repodirext)-1)
		self.datasetrng = self.datasetsize
                
		self.timeseries_repo = []
        
		for i in range(len(self.repodirext)):			
			"""
			The following two lines are used instead of the third line when DataMarket is the data source.
			"""
			# ts = pd.read_csv(self.repodirext[i], usecols=[1], header=0, skipfooter=2, names=['value'], engine='python')
			# ts['anomaly'] = pd.Series(np.zeros(len(ts['value'])), index=ts.index)
			
			"""
			The following line is used instead of the third line when Numenta is the data source.
			"""
			ts = pd.read_csv(self.repodirext[i], usecols=[1,3], header=0, names=['value','anomaly'])
			
			"""
			The following line is used instead of the third line when Yahoo Benchmark is the data source.
			"""			
			# ts = pd.read_csv(self.repodirext[i], usecols=[1,2], header=0, names=['value','anomaly'])
			
			ts = ts.astype(np.float32)
			
			scaler = sklearn.preprocessing.MinMaxScaler()
			scaler.fit(np.array(ts['value']).reshape(-1,1))
			ts['value'] = scaler.transform(np.array(ts['value']).reshape(-1,1))
			
			self.timeseries_repo.append(ts)
    
    # reset the instance
    def reset(self):
        # 1. select a new time series from the repo and load
        # the time series contains "timestamp", "value", "anomaly"
        if self.datasetfix == 0:        
	        self.datasetidx = (self.datasetidx + 1) % self.datasetrng
	        
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init

        # 2. return the first state, containing the first element of the time series
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        
        return self.timeseries_states
        
    # return the whole dataset
    def reset_getall(self):
        # 1. select a new time series from the repo and load
        # the time series contains "timestamp", "value", "anomaly"
        if self.datasetfix == 0:        
	        self.datasetidx = (self.datasetidx + 1) % self.datasetrng
	        
        self.timeseries = pd.read_csv(self.repodirext[self.datasetidx], \
        							usecols=[0,1,2], header=0, names=['timestamp','value','anomaly'])       
        self.timeseries = self.timeseries.astype(np.float32)
        self.timeseries_curser = self.timeseries_curser_init

        # 2. Preprocess the time series values
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(np.array(self.timeseries['value']).reshape(-1,1))
        self.timeseries['value'] = scaler.transform(np.array(self.timeseries['value']).reshape(-1,1))
        
        return self.timeseries
        
    # take a step and gain a reward
    def step(self, action):
        # assert(action in action_space)
        # assert(self.timeseries_curser >= 0)
        
        # 1. get the reward of the action
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        
        # 2. get the next state and the done flag after the action
        self.timeseries_curser += 1
        
        if self.timeseries_curser >= self.timeseries['value'].size:
        	done = 1
        	state = np.array([self.timeseries_states, self.timeseries_states])
        else:
        	done = 0
        	state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)
        
        if len(np.shape(state)) > len(np.shape(self.timeseries_states)):
        	self.timeseries_states = state[action]
        else:
        	self.timeseries_states = state
            
        return state, reward, done, []