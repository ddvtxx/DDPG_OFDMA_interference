import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import math
#from google.colab import files 
import environment_simulation_move as env
from ReplayBuffer import ReplayBuffer,create_directory
from DDPG_agent import DDPG
import random
print(T.__version__)

for i_loop in range(1):
    #can only deal with 10 users per ap at most
    numAPuser = 5
    numRU = 8
    numSenario = 1
    linkmode = 'uplink'
    ru_mode = 4
    episode = 600
    max_iteration = 200
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

    store_loss = False
    store_detail = True

    reward_history = []
    system_bitrate_history = []
    reward_ave_history = []
    system_ave_bitrate_history = []

    actor_loss_history = []
    critic_loss_history = []

    detail_bitrate_history = []

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

    for i_episode in range(20, episode):
        test_env.change_RU_mode(4)
        x_init,y_init = test_env.senario_user_local_init()
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        channel_gain_obs = test_env.channel_gain_calculate()
        RU_mapper = test_env.n_AP_RU_mapper()
        system_bitrate = test_env.calculate_4_cells(RU_mapper)
        observation = test_env.get_sinr()
        # system_bitrate_history.append(system_bitrate)
        test_env.change_RU_mode(3)
        for i_iteration in range(max_iteration):
            action_array = []

            for i_agent in range(4):
                DDPG_agent = agent_array[i_agent]
                action_pre = DDPG_agent.choose_action(observation[i_agent].reshape((1, numAPuser, numRU)),train=True)
                user_list = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,0,0,0]
                action_pre = action_pre.reshape(numAPuser,numRU)
                # action_0 = action_pre
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
            system_bitrate = test_env.calculate_4_cells(RU_mapper)
            observation_ = test_env.get_sinr()
            key_value = system_bitrate/(1e+6)
            if i_episode%20 == 0 and store_detail:
                detail_bitrate_history.append(system_bitrate)
            
            reward = key_value
            reward = np.zeros((4))
            for i_reward in range(4):
                # reward[i_reward] = (system_bitrate/10 - RU_mapper[i_reward].sum()*test_env.bwRU)/(1e+4)-1100
                reward[i_reward] = key_value

            
            x_, y_ = test_env.senario_user_local_move(x,y)
            userinfo_ = test_env.senario_user_info(x_,y_)
            channel_gain_obs_ = test_env.channel_gain_calculate()

            for i_agent in range(4):
                action = action_array[i_agent]
                DDPG_agent.remember(observation[i_agent].reshape((1, numAPuser, numRU)), action, reward[i_agent], observation_[i_agent].reshape((1, numAPuser, numRU)), done=False)
                DDPG_agent.learn()
                if i_episode%20 == 0 and store_loss:
                    critic_loss = DDPG_agent.get_critic_loss()
                    actor_loss = DDPG_agent.get_actor_loss()
                    critic_loss_history.append(critic_loss)
                    actor_loss_history.append(actor_loss)                    

            observation = observation_
            x, y= x_, y_

            system_bitrate_history.append(system_bitrate)
            reward_history.append(reward)

            if i_iteration == max_iteration-1:
                reward_ave = np.mean(reward_history)
                system_bitrate_ave = np.mean(system_bitrate_history)
                reward_history = []
                system_bitrate_history = []
                reward_ave_history.append(reward_ave)
                system_ave_bitrate_history.append(system_bitrate_ave)
                print('i_loop =', i_loop,'i_episode =',i_episode, 'reward =',reward_ave, 'system_bitrate =',system_bitrate_ave)
                if i_episode%20 == 0 and store_loss:
                    dataframe=pd.DataFrame({'critic_loss':critic_loss_history})
                    dataframe.to_csv("./result/critic_loss_"+str(i_episode)+".csv", index=False,sep=',')
                    dataframe=pd.DataFrame({'actor_loss':actor_loss_history})
                    dataframe.to_csv("./result/actor_loss_"+str(i_episode)+".csv", index=False,sep=',')
                    critic_loss_history = []
                    actor_loss_history = []
                if i_episode%20 == 0 and store_detail:
                    dataframe=pd.DataFrame({'critic_loss':detail_bitrate_history})
                    dataframe.to_csv("./result/detail_bitrate_"+str(i_episode)+".csv", index=False,sep=',')
                    detail_bitrate_history = []

        dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
        dataframe.to_csv("./result/bitrate_multiple_sinr_"+str(i_loop)+".csv", index=False,sep=',')