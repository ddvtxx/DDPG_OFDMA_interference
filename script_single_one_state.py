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

#can only deal with 10 users per ap at most
numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 3
# episode = 2
episode = 200*600
DDPG_agent = DDPG(alpha=1e-4, beta=2e-4,numSenario=numSenario,numAPuser=numAPuser,numRU=numRU,
      actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
      actor_fc4_dim=2**6,
      critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
      critic_fc4_dim=2**6,
      ckpt_dir='./DDPG/',
      gamma=0,tau=0.001,action_noise=1e-5,max_size=100000,batch_size=128)
create_directory('./DDPG/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
# reward_history = []
# system_bitrate_history = []
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
for i_episode in range(episode):
    test_env.change_random_seed(i_episode)
    x,y = test_env.senario_user_local_init()
    userinfo = test_env.senario_user_info(x,y)
    channel_gain_obs = test_env.channel_gain_calculate()
    observation = channel_gain_obs[3,:,:]
    AP123_RU_mapper = test_env.n_AP_RU_mapper()
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
          action_pre[max_key,:] = 0
          max_key = np.argmax(action_pre[:,k])
          if max_key not in user_list:
            action_pre[max_key,:] = 0
            max_key = np.argmax(action_pre[:,k])      
            user_list.remove(max_key)
            action_0[max_key,k] = 1
          user_list.remove(max_key)
          action_0[max_key,k] = 1
    action_1=action_0.reshape(1,numAPuser,numRU)
    RU_mapper = np.vstack((AP123_RU_mapper,action_1))
    system_bitrate = test_env.calculate_4_cells(RU_mapper)
    key_value = system_bitrate/(1e+4)
    reward = key_value
    done = False
    DDPG_agent.remember(observation, action_0, reward, observation, done)
    DDPG_agent.learn()
    # system_bitrate_history.append(system_bitrate)
    # reward_history.append(reward)
    print('i_episode =',i_episode, 'reward =',reward, 'system_bitrate =',system_bitrate)
    # if i_episode == episode-1:
    #     dataframe=pd.DataFrame({'bitrate':system_bitrate_history})
    #     dataframe.to_csv("./result/bitrate_single_one_state.csv", index=False,sep=',')
reward_array = []
for i_test in range(100):
  random.seed(0)
  random_number = random.randint(0, 119999)
  test_env.change_random_seed(random_number)
  x,y = test_env.senario_user_local_init()
  userinfo = test_env.senario_user_info(x,y)
  channel_gain_obs = test_env.channel_gain_calculate()
  observation = channel_gain_obs[3,:,:]
  AP123_RU_mapper = test_env.n_AP_RU_mapper()
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
        action_pre[max_key,:] = 0
        max_key = np.argmax(action_pre[:,k])
        if max_key not in user_list:
          action_pre[max_key,:] = 0
          max_key = np.argmax(action_pre[:,k])      
          user_list.remove(max_key)
          action_0[max_key,k] = 1
        user_list.remove(max_key)
        action_0[max_key,k] = 1
  action_1=action_0.reshape(1,numAPuser,numRU)
  RU_mapper = np.vstack((AP123_RU_mapper,action_1))
  system_bitrate = test_env.calculate_4_cells(RU_mapper)
  key_value = system_bitrate/(1e+4)
  reward = key_value
  reward_array.append(reward)
print("The test result is that ",np.mean(reward_array))