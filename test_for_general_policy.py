import environment_simulation_move as env
import numpy as np

test_env = env.environment_base(5,8,'uplink',3)
x_init,y_init = test_env.senario_user_local_init()
x,y = x_init,y_init
userinfo = test_env.senario_user_info(x,y)
channel_gain_obs = test_env.channel_gain_calculate()
AP123_RU_mapper = test_env.n_AP_RU_mapper()
general_mapper = np.zeros((5,8))
general_mapper[0,3]=1
general_mapper[1,5]=1
general_mapper[2,7]=1
general_mapper[3,1]=1
general_mapper[3,6]=1
general_mapper[4,0]=1
general_mapper[4,2]=1
general_mapper[4,4]=1
# general_mapper = [[0, 0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0, 1],
#                     [0, 1, 0, 0, 0, 0, 1, 0],
#                     [1, 0, 1, 0, 1, 0, 0, 0]]

RU_mapper = np.vstack((AP123_RU_mapper, general_mapper.reshape(1,5,8)))
bitrate = test_env.calculate_4_cells(RU_mapper)
dis = test_env.cal_user_dis(x,y)

# print("For the "+str(3)+" AP.")
# for j in range(5):
#     print(str(int(x[3,j]))+","+str(int(y[3,j])))
#     print(dis[3,j])
#     print(test_env.channel_gain[3,3])
#     print("")
print(bitrate)

#分配RU时，考虑干扰！