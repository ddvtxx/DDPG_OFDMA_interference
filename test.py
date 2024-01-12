import environment_simulation_move as env
import numpy as np

numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 4
episode = 2000
max_iteration = 200
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
x_init,y_init = test_env.senario_user_local_init()
x,y = x_init,y_init
userinfo = test_env.senario_user_info(x,y)
channel_gain_obs = test_env.channel_gain_calculate()
#(ap,ap,user,ru)
ru_mapper = test_env.n_AP_RU_mapper()
system_bitrate = test_env.calculate_4_cells(ru_mapper)
ru_per_user = test_env.get_ru_per_user()

print(system_bitrate/10-ru_mapper[0].sum()*test_env.bwRU)