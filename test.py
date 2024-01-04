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

#(ap,user,ru)
observation_array=[]
for i_observation in range(4):
    #select the channel gain as the state
    # observation = channel_gain_obs[i_observation,:,:]
    # observation_array.append(observation)
    #select the possible interference as the state
    #(ap,ap,user,ru)
    observation = np.zeros((numAPuser,numRU))
    for i_user in range(numAPuser):
        for i_ru in range(numRU):
            observation[i_user, i_ru] = channel_gain_obs[:,:,:,i_ru].sum(axis=0).sum(axis=0).sum(axis=0) - channel_gain_obs[i_observation,i_observation,i_user,i_ru]
    observation_array.append(observation)

print(observation_array[1])
print(channel_gain_obs[1])
