import environment_simulation_move as env
import numpy as np

numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 3
episode = 2000
max_iteration = 200
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
x_init,y_init = test_env.senario_user_local_init()
x,y = x_init,y_init
userinfo = test_env.senario_user_info(x,y)
#(ap,ap,user,ru)
channel_gain_obs = test_env.channel_gain_calculate()
mapper_1 = test_env.n_AP_RU_mapper()
mapper_2 = test_env.n_AP_RU_mapper()
mapper_3 = test_env.n_AP_RU_mapper()
ru_mapper = np.vstack((mapper_1, mapper_2, mapper_3))

system_bitrate = test_env.calculate_4_cells(ru_mapper)

print(system_bitrate)

