import environment_simulation_move as env
import numpy as np
import pandas as pd

for i_seed in range(1,16):
    numAPuser = 5
    numRU = 8
    numSenario = 4
    linkmode = 'uplink'
    ru_mode = 4
    episode = 600
    max_iteration = 200
    system_bitrate_history = []
    system_ave_bitrate_history = []
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode,seed=i_seed)
    x,y = test_env.senario_user_local_init()

    for i_iteration in range(max_iteration):
        userinfo = test_env.senario_user_info(x,y)
        #(ap,ap,user,ru)
        channel_gain_obs = test_env.channel_gain_calculate()
        ru_mapper = test_env.n_AP_RU_mapper()
        bitrate = test_env.calculate_4_cells(ru_mapper)
        system_bitrate_history.append(bitrate)
        x_init,y_init = test_env.senario_user_local_move(x,y)
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        #(ap,ap,user,ru)
        channel_gain_obs = test_env.channel_gain_calculate()
    bitrate_ave = np.mean(system_bitrate_history)
    print(bitrate_ave)
    for i in range(600):
        system_ave_bitrate_history.append(bitrate_ave)
    
    dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
    dataframe.to_csv("./result/bitrate_general_wf_seed_"+str(i_seed)+".csv", index=False,sep=',')