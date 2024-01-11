# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import math
import environment_simulation_move as env
from ReplayBuffer import ReplayBuffer,create_directory
from DDPG_agent import DDPG
import random
print(T.__version__)

#can only deal with 10 users per ap at most
numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 3
episode = 600
max_iteration = 200
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

reward_history = []
system_bitrate_history = []
reward_ave_history = []
system_ave_bitrate_history = []

agent_array = []
#create four agents for four APs
for i in range(4):
    DDPG_agent=globals()[f"DDPG_agent_{i}"] = DDPG(
        alpha=1e-4, beta=2e-4,numSenario=numSenario,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./DDPG'+str(i)+'/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
    agent_array.append(DDPG_agent)
    create_directory('./DDPG_'+str(i)+'/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

#episode start
for i_episode in range(episode):
    x_init,y_init = test_env.senario_user_local_init()
    x,y = x_init,y_init
    userinfo = test_env.senario_user_info(x,y)
    channel_gain_obs = test_env.channel_gain_calculate()

    #(ap,user,ru)
    observation_array=[]
    for i_observation in range(4):
        #select the channel gain as the state
        observation = channel_gain_obs[i_observation,:,:]
        observation_array.append(observation)

    #iteration start
    for i in range(max_iteration):
        action_array = []

        for i_agent in range(4):
            DDPG_agent = agent_array[i_agent]
            observation = observation_array[i_agent]
            done = False
            action_pre = DDPG_agent.choose_action(observation,train=True)
            user_list = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,0,0,0]
            action_pre = action_pre.reshape(numAPuser,numRU)
            action_0 = np.zeros_like(action_pre)
            for k in range(numRU):
                max_key = np.argmax(action_pre[:,k])
                if numAPuser <=2:
                    if max_key in user_list:
                        action_0[max_key,k] = 1
                        user_list.remove(max_key)   
                else:       
                    if max_key in user_list:
                        action_0[max_key,k] = 1
                        user_list.remove(max_key)
                    else:
                        action_pre[max_key,:] = 0
                        max_key = np.argmax(action_pre[:,k])
                        if max_key not in user_list:
                            action_pre[max_key,:] = 0
                            max_key = np.argmax(action_pre[:,k])      
                            user_list.remove(max_key)
                            action_0[max_key,k] = 1
                        user_list.remove(max_key)
                        action_0[max_key,k] = 1
            action_1 = action_0.reshape(1, numAPuser, numRU)
            action_array.append(action_0)
        #only work for 4-agent
        RU_mapper = np.vstack((action_array[0].reshape(1,numAPuser,numRU), action_array[1].reshape(1,numAPuser,numRU), action_array[2].reshape(1,numAPuser,numRU), action_array[3].reshape(1,numAPuser,numRU)))
        system_bitrate , sinr= test_env.calculate_4_cells(RU_mapper)
        key_value = system_bitrate/(1e+6)
        reward = key_value
        x_, y_ = test_env.senario_user_local_move(x,y)
        userinfo_ = test_env.senario_user_info(x_,y_)
        channel_gain_obs_ = test_env.channel_gain_calculate()

        observation_array_ = []
        for index in range(4):
            observation_ = channel_gain_obs_[index,:,:]

            observation_array_.append(observation_)
            DDPG_agent = agent_array[index]

            state = observation_array[index]
            action = action_array[index]
            state_ = observation_array_[index]

            DDPG_agent.remember(state, action, reward, state_, done)
            DDPG_agent.learn()
        observation_array = observation_array_
        x, y= x_, y_
 
        system_bitrate_history.append(system_bitrate)
        reward_history.append(reward)
        if i == max_iteration-1:
            reward_ave = np.mean(reward_history)
            system_bitrate_ave = np.mean(system_bitrate_history)
            reward_history = []
            system_bitrate_history = []
            reward_ave_history.append(reward_ave)
            system_ave_bitrate_history.append(system_bitrate_ave)
            print('i_episode =',i_episode, 'reward =',reward_ave, 'system_bitrate =',system_bitrate_ave)

dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
dataframe.to_csv("./result/bitrate_multiple_wf.csv", index=False,sep=',')