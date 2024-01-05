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
channel_gain_obs = test_env.channel_gain_calculate()
#(ap,ap,user,ru)
ru_mapper = test_env.n_AP_RU_mapper()
strength_tem = channel_gain_obs[0][0] * ru_mapper
strength_sum = strength_tem.sum(axis=2)
# print(range(channel_gain_obs.shape[0]))
# for strength_total in strength_sum:
#     for i in range():

print(strength_tem)

