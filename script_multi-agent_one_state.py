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
episode = 200*600
# episode = 600
# max_iteration = 200

agent_array = []
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