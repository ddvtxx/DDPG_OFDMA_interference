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
        