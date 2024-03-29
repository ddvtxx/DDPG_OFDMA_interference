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

for i_loop in range(6):
    #can only deal with 10 users per ap at most
    numAPuser = 5
    numRU = 8
    numSenario = 4
    linkmode = 'uplink'
    ru_mode = 3
    episode = 600
    max_iteration = 200
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
    DDPG_agent = DDPG(alpha=1e-4, beta=2e-4,numSenario=numSenario,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./DDPG/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
    create_directory('./DDPG/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
    
    reward_history = []
    system_bitrate_history = []
    reward_ave_history = []
    system_ave_bitrate_history = []

    for i_episode in range(episode):
        actor_loss_history = []
        critic_loss_history = []
        test_env.change_RU_mode(4)
        x_init,y_init = test_env.senario_user_local_init()
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        channel_gain_obs = test_env.channel_gain_calculate()
        RU_mapper = test_env.n_AP_RU_mapper()
        system_bitrate = test_env.calculate_4_cells(RU_mapper)
        observation = test_env.get_sinr()
        system_bitrate_history.append(system_bitrate)
        test_env.change_RU_mode(3)
        for i_iteration in range(max_iteration):
            action_pre = DDPG_agent.choose_action(observation,train=False)
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
                        while max_key not in user_list:
                            action_pre[max_key, :] = 0
                            max_key = np.argmax(action_pre[:,k])
                        action_0[max_key, k] = 1
                        user_list.remove(max_key)
                        # action_pre[max_key,:] = 0
                        # max_key = np.argmax(action_pre[:,k])
                        # if max_key not in user_list:
                        #     action_pre[max_key,:] = 0
                        #     max_key = np.argmax(action_pre[:,k])      
                        #     user_list.remove(max_key)
                        #     action_0[max_key,k] = 1
                        # user_list.remove(max_key)
                        # action_0[max_key,k] = 1
            action_1=action_0.reshape(1,numAPuser,numRU)
            AP123_RU_mapper = test_env.n_AP_RU_mapper()
            RU_mapper = np.vstack((AP123_RU_mapper,action_1))
            system_bitrate = test_env.calculate_4_cells(RU_mapper)
            observation_ = test_env.get_sinr()

            individual_bitrate = test_env.get_n_AP_n_user_bitrate()

            key_value = individual_bitrate[3]
            reward = key_value

            x_,y_ = test_env.senario_user_local_move(x,y)
            userinfo_ = test_env.senario_user_info(x_,y_)
            channel_gain_obs_ = test_env.channel_gain_calculate()
            done = False
            DDPG_agent.remember(observation[3], action_0, reward, observation_[3], done)
            DDPG_agent.learn()
            actor_loss = DDPG_agent.get_actor_loss()
            critic_loss = DDPG_agent.get_actor_loss()
            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)

            observation = observation_
            x,y=x_,y_

            system_bitrate_history.append(system_bitrate)
            reward_history.append(reward)
        
            if i_iteration == max_iteration-1:
                reward_ave = np.mean(reward_history)
                system_bitrate_ave = np.mean(system_bitrate_history)
                reward_history = []
                system_bitrate_history = []
                reward_ave_history.append(reward_ave)
                system_ave_bitrate_history.append(system_bitrate_ave)
                print('i_loop =', i_loop, 'i_episode =',i_episode, 'reward =',reward_ave, 'system_bitrate =',system_bitrate_ave)
        
        if i_episode % 50 == 0 and i_loop%2 == 0:
            dataframe=pd.DataFrame({'bitrate':actor_loss_history})
            dataframe.to_csv("./result/actor_loss_sinr_single_individual_loop"+str(i_loop)+"_epis"+str(i_episode)+".csv", index=False,sep=',')
            dataframe=pd.DataFrame({'bitrate':critic_loss_history})
            dataframe.to_csv("./result/critic_loss_sinr_single_individual_loop"+str(i_loop)+"_epis"+str(i_episode)+".csv", index=False,sep=',')

    dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
    dataframe.to_csv("./result/bitrate_sinr_single_individual_"+str(i_loop)+".csv", index=False,sep=',')
