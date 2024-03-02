import numpy as np
import pandas as pd
import environment_simulation_move as env

numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 3
episode = 600
max_iteration = 200
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode,seed=9)
system_bitrate_history = []
system_ave_bitrate_history = []
for i_iteration in range(max_iteration):
    x_init,y_init = test_env.senario_user_local_init()
    x,y = x_init,y_init
    userinfo = test_env.senario_user_info(x,y)
    channel_gain_obs = test_env.channel_gain_calculate()
    ru_map = np.zeros((4,5,8))
    for i_ap in range(numSenario):
        for i_user in range(numAPuser):
            for i_ru in range(numRU):
                ru_map[i_ap][int(np.random.random()*4)][i_ru] = 1
    system_bitrate = test_env.calculate_4_cells_without_wf(ru_map)
    x_,y_ = test_env.senario_user_local_move(x,y)
    x, y =x_, y_
    system_bitrate_history.append(system_bitrate)

    if i_iteration == max_iteration-1:
        system_bitrate_ave = np.mean(system_bitrate_history)
        system_bitrate_history = []
        # system_ave_bitrate_history.append(system_bitrate_ave)
        print('system_bitrate =',system_bitrate_ave)
        for i_episode in range(episode):
            system_ave_bitrate_history.append(system_bitrate_ave)
dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
dataframe.to_csv("E:/FYP/Modification Code/DDPG_OFDMA_interference/result/random_bitrate_seed_9.csv", index=False,sep=',')
