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

for i_loop in range(6):
    #can only deal with 10 users per ap at most
    numAPuser = 5
    numRU = 8
    numSenario = 4
    linkmode = 'uplink'
    ru_mode = 3
    episode = 10000
    max_iteration = 200
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

    DDPG_agent_s = DDPG(alpha=1e-4, beta=2e-4,numSenario=1,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./DDPG_s/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
    create_directory('./DDPG_s/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    DDPG_agent_m = DDPG(alpha=1e-4, beta=2e-4,numSenario=3,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./DDPG_m/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
    create_directory('./DDPG_m/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])


    reward_history = []
    system_bitrate_history = []
    reward_ave_history = []
    system_ave_bitrate_history = []
    
    for i_episode in range(episode):
        actor_loss_history_s = []
        critic_loss_history_s = []
        actor_loss_history_m = []
        critic_loss_history_m = []
        test_env.change_RU_mode(4)
        x_init,y_init = test_env.senario_user_local_init()
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        channel_gain_obs = test_env.channel_gain_calculate()
        test_env.change_RU_mode(4)
        general_mapper = test_env.n_AP_RU_mapper()
        general_bitrate = test_env.calculate_4_cells(general_mapper)
        test_env.change_RU_mode(3)
        observation = test_env.get_sinr()
        test_env.change_RU_mode(3)

        for i_iteration in range(max_iteration):
            action_pre_s = DDPG_agent_s.choose_action(observation[0], train=False)
            action_pre_m = DDPG_agent_m.choose_action(observation[1:], train=False)
            action_pre_s = action_pre_s.reshape(1, numAPuser, numRU)
            action_pre_m = action_pre_m.reshape(3, numAPuser, numRU)
            action_pre = np.vstack((action_pre_s, action_pre_m))
            action_0 = np.zeros_like(action_pre)
            user_resource_count = np.zeros((action_pre.shape[0], action_pre.shape[1]), dtype=int)
            for m in range(numSenario):
                for resource_index in range(action_pre.shape[2]):
                    user_index = np.argmax(action_pre[m,:,resource_index])
                    if user_resource_count[m,user_index] < 3:
                        action_0[m,user_index,resource_index] = 1
                        user_resource_count[m,user_index] += 1

            system_bitrate = test_env.calculate_4_cells(action_0)
            observation_ = test_env.get_sinr()

            key_value = system_bitrate/(1e+4)
            reward = key_value
            x_,y_ = test_env.senario_user_local_move(x,y)
            userinfo_ = test_env.senario_user_info(x_,y_)
            channel_gain_obs_ = test_env.channel_gain_calculate()

            done = False
            DDPG_agent_s.remember(observation[0], action_0[0], reward, observation_[0], done)
            DDPG_agent_s.learn()
            DDPG_agent_m.remember(observation[1:], action_0[1:], reward,observation_[1:], done)
            DDPG_agent_m.learn()

            actor_loss_s = DDPG_agent_s.get_actor_loss()
            critic_loss_s = DDPG_agent_s.get_critic_loss()
            actor_loss_history_s.append(actor_loss_s)
            critic_loss_history_s.append(critic_loss_s)

            actor_loss_m = DDPG_agent_m.get_actor_loss()
            critic_loss_m = DDPG_agent_m.get_critic_loss()
            actor_loss_history_m.append(actor_loss_m)
            critic_loss_history_m.append(critic_loss_history_m)

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
            dataframe=pd.DataFrame({'bitrate':actor_loss_history_s})
            dataframe.to_csv("./result/DDPG_actor_loss_sinr_mix_s_loop"+str(i_loop)+"_epis"+str(i_episode)+".csv", index=False,sep=',')
            dataframe=pd.DataFrame({'bitrate':critic_loss_history_s})
            dataframe.to_csv("./result/DDPG_critic_loss_sinr_mix_s_loop"+str(i_loop)+"_epis"+str(i_episode)+".csv", index=False,sep=',')
            dataframe=pd.DataFrame({'bitrate':actor_loss_history_m})
            dataframe.to_csv("./result/DDPG_actor_loss_sinr_mix_m_loop"+str(i_loop)+"_epis"+str(i_episode)+".csv", index=False,sep=',')
            dataframe=pd.DataFrame({'bitrate':critic_loss_history_m})
            dataframe.to_csv("./result/DDPG_critic_loss_sinr_mix_m_loop"+str(i_loop)+"_epis"+str(i_episode)+".csv", index=False,sep=',')

    dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
    dataframe.to_csv("./result/DDPG_bitrate_sinr_mix_"+str(i_loop)+".csv", index=False,sep=',')