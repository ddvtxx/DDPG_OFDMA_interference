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
test_env.change_RU_mode(4)
system_bitrate_history = []
system_ave_bitrate_history = []
for i_iteration in range(max_iteration):
    x_init,y_init = test_env.senario_user_local_init()
    x,y = x_init,y_init
    userinfo = test_env.senario_user_info(x,y)
    channel_gain_obs = test_env.channel_gain_calculate()
    RU_mapper = test_env.n_AP_RU_mapper()
    system_bitrate = test_env.calculate_4_cells_without_wf(RU_mapper)
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
dataframe.to_csv("E:/FYP/Modification Code/DDPG_OFDMA_interference/result/general_bitrate_seed_9.csv", index=False,sep=',')
