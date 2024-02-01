import environment_simulation_move as env
import numpy as np

# numAPuser = 5
# numRU = 8
# numSenario = 4
# linkmode = 'uplink'
# ru_mode = 4
# episode = 2000
# max_iteration = 200
# test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
# x_init,y_init = test_env.senario_user_local_init()
# x,y = x_init,y_init
# userinfo = test_env.senario_user_info(x,y)
# channel_gain_obs = test_env.channel_gain_calculate()
# #(ap,ap,user,ru)
# ru_mapper = test_env.n_AP_RU_mapper()
# system_bitrate = test_env.calculate_4_cells(ru_mapper)
# # ru_per_user = test_env.get_ru_per_user()

# print(system_bitrate/10-ru_mapper[0].sum()*test_env.bwRU)

import numpy as np

def allocate_RUs(reward_matrix):
    num_people, num_RUs = reward_matrix.shape
    # 创建一个同样大小的矩阵，用于标记球是否已被分配
    allocated = np.zeros(num_RUs, dtype=bool)
    # 创建一个数组，用于记录每个人拿到的球数
    RUs_per_user = np.zeros(num_people, dtype=int)
    # 创建结果列表，记录每个人分配到的球
    allocation = [[] for _ in range(num_people)]
    # 按奖励大小排序每个人的球的索引
    sorted_indices = np.argsort(-reward_matrix, axis=1)

    # 先确保每个人至少分到一个球
    for user in range(num_people):
        for RU_index in sorted_indices[user]:
            if not allocated[RU_index]:
                allocation[user].append(RU_index)
                allocated[RU_index] = True
                RUs_per_user[user] += 1
                break

    # 然后根据奖励继续分配剩下的球，直到每个人最多有三个球
    for user in range(num_people):
        for RU_index in sorted_indices[user]:
            if not allocated[RU_index] and RUs_per_user[user] < 3:
                allocation[user].append(RU_index)
                allocated[RU_index] = True
                RUs_per_user[user] += 1

    # 将分配结果转换为所要求的格式
    result_matrix = np.zeros((num_people, num_RUs), dtype=int)
    for user, RUs in enumerate(allocation):
        for RU in RUs:
            result_matrix[user, RU] = 1

    return result_matrix

# 示例奖励矩阵
reward_matrix = np.random.rand(5, 8)

# 执行分配函数
result_matrix = allocate_RUs(reward_matrix)
print("Reward Matrix:\n", reward_matrix)
print("Allocation Matrix:\n", result_matrix)