# -*- coding: utf-8 -*-
"""
Created on Wed May  9 17:12:27 2018

@author: Hussein
"""
## source 
## "Further Optimal Regret Bounds for Thompson Sampling Shipra"
## by Shipra Agrawal and Navin Goyal 

# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset1 = pd.read_csv('cc.csv')

# Implementing Thompson Sampling

N = 10000
d = 10
ads_selected = []
mu = [0] * d
k_i = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

reward_all = np.zeros((0,10))
std = 0
for i in range(0,d):
    ads_selected.append(i)
    reward = dataset1.values[0, i]
    sums_of_rewards[i] = sums_of_rewards[i] + reward
    total_reward = total_reward + reward
    #mu[ad] = (mu[ad] * k_i[ad] + reward) / (k_i[ad] + 2)
    mu[i] = (sums_of_rewards[i]) / (k_i[i] + 1)
    k_i[i] = k_i[i] + 1
    #print ("-------------")
    
for n in range(1, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_gauss = random.normalvariate(mu[i], k_i[i]+1)
       # print (random_gauss)
        if random_gauss > max_random:
            max_random = random_gauss
            ad = i
    ads_selected.append(ad)
    reward = dataset1.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    mu[ad] = (mu[ad] * k_i[ad] + reward) / (k_i[ad] + 2)
    #mu[ad] = (sums_of_rewards[ad]) / (k_i[ad] + 1)
    k_i[ad] = k_i[ad] + 1
    #print ("-------------")





























