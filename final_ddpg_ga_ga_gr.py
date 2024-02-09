#  ddpg global agent global action global reward

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
        numSenario = 4
        linkmode = 'uplink'
        ru_mode = 3
        episode = 2000
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
            system_bitrate = test_env.calculate_4_cells_without_wf(RU_mapper)
            observation = test_env.get_sinr()
            test_env.change_RU_mode(3)
            for i_iteration in range(max_iteration):
                action_pre = DDPG_agent.choose_action(observation,train=False)
                action_pre = action_pre.reshape(numSenario,numAPuser,numRU)