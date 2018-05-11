# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:36:09 2018

@author: Hussein
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats as ss

import math
dataset1 = pd.read_csv('cc.csv')

# Implementing Thompson Sampling
## source
## "DECISION MAKING USING THOMPSON SAMPLING" 
## by Joseph Mellor School. 
## page 69
N = 10000
d = 10
ads_selected = []
mu = [0] * d
s = [1] * d

sums_of_rewards = [0] * d
total_reward = 0

reward_all = np.zeros((0,10))

for i in range(0,d):
    ads_selected.append(i)
    reward = dataset1.values[0, i]
    sums_of_rewards[i] = sums_of_rewards[i] + reward
    total_reward = total_reward + reward
    row = np.full((1,10),-1)
    row[0,i] = reward
    reward_all = np.vstack([reward_all,row[0]])
    mu[i] = reward
    s[i] = 1
    
    
for n in range(1, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_gauss = random.normalvariate(mu[i], s[i])
        
        #print (random_gauss)
        if random_gauss > max_random:
            max_random = random_gauss
            ad = i
    #print ("_-------------------------")        
    ads_selected.append(ad)
    reward = dataset1.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    
    total_reward = total_reward + reward
    row = np.full((1,10),-1)
    row[0,ad] = reward
    reward_all = np.vstack([reward_all,row[0]])
    
    #mu[ad] = (mu[ad] * k_i[ad] + reward) / (k_i[ad] + 2)
    std_2 = math.pow(np.std(reward_all[:,ad],ddof=1),2)
    
    mu[ad] = ((mu[ad] * std_2) + (reward * s[ad])) / (std_2 + s[ad])
    s[ad] = (s[ad] * std_2) / (s[ad] * std_2)
    
    
    
sum = np.array(dataset1).sum(axis=0)