# multiple agent individual action global reward

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

for i_seed in range(50):
    for i_loop in range(4):
        numAPuser = 5
        numRU = 8
        numSenario = 1
        linkmode = 'uplink'
        ru_mode = 4
        episode = 600
        max_iteration = 200
        test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode,seed=i_seed)
        agent_array = []
        for i_agent in range(4):
            DDPG_agent = DDPG(alpha=1e-4, beta=2e-4,numSenario=numSenario,numAPuser=numAPuser,numRU=numRU,
                actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
                actor_fc4_dim=2**6,
                critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
                critic_fc4_dim=2**6,
                ckpt_dir='./DDPG_'+str(i_agent)+'/',
                gamma=0.99,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
            create_directory('./DDPG_'+str(i_agent)+'/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
            agent_array.append(DDPG_agent)
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
            system_bitrate = test_env.calculate_4_cells_without_wf(RU_mapper)
            observation = test_env.get_sinr()
            system_bitrate_history.append(system_bitrate)
            test_env.change_RU_mode(3)
            for i_iteration in range(max_iteration):
                RU_mapper = np.zeros((4, numAPuser, numRU))
                for i_agent in range(4):
                    DDPG_agent = agent_array[i_agent]
                    action_pre = DDPG_agent.choose_action(observation[i_agent], train=False)
                    action_pre = action_pre.reshape(numAPuser,numRU)
                    action_0 = test_env.allocate_RUs(action_pre)
                    RU_mapper[i_agent,:,:] = action_0
                system_bitrate = test_env.calculate_4_cells_without_wf(RU_mapper)
                observation_ = test_env.get_sinr()
                key_value = system_bitrate/(1e+4)
                reward = key_value
                x_,y_ = test_env.senario_user_local_move(x,y)
                userinfo_ = test_env.senario_user_info(x_,y_)
                channel_gain_obs_ = test_env.channel_gain_calculate()
                done = False
                actor_loss = 0
                critic_loss = 0
                for i_agent in range(4):
                    DDPG_agent = agent_array[i_agent]
                    DDPG_agent.remember(observation[i_agent], RU_mapper[i_agent], reward, observation_[i_agent], done)
                    actor_loss += DDPG_agent.get_actor_loss()
                    critic_loss += DDPG_agent.get_critic_loss()
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
                    print('i_seed =',i_seed,'i_loop =', i_loop, 'i_episode =',i_episode, 'reward =',reward_ave, 'system_bitrate =',system_bitrate_ave)
                    

            if (i_episode+1) % 50 == 0 or i_episode == 0:
                dataframe=pd.DataFrame({'bitrate':actor_loss_history})
                dataframe.to_csv("E:/FYP/Modification Code/DDPG_OFDMA_interference/result/ddpg_actor_loss_mua_ia_gr_seed_"+str(i_seed)+"_loop_"+str(i_loop)+"_episode_"+str(i_episode)+".csv", index=False,sep=',')
                dataframe=pd.DataFrame({'bitrate':critic_loss_history})
                dataframe.to_csv("E:/FYP/Modification Code/DDPG_OFDMA_interference/result/ddpg_critic_loss_mua_ia_gr_seed_"+str(i_seed)+"_loop_"+str(i_loop)+"_episode_"+str(i_episode)+".csv", index=False,sep=',')
 
        dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
        dataframe.to_csv("E:/FYP/Modification Code/DDPG_OFDMA_interference/result/ddpg_bitrate_mua_ia_gr_seed_"+str(i_seed)+"_loop_"+str(i_loop)+".csv", index=False,sep=',')