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

# import numpy as np

# def allocate_RUs(single_ru_mapper):
#     num_people, num_RUs = single_ru_mapper.shape
#     # 标记数组，记录每个球是否已分配
#     allocated = np.zeros(num_RUs, dtype=bool)
#     # 记录每个人已分配球的数量
#     RUs_per_user = np.zeros(num_people, dtype=int)
#     # 记录每个人的球分配情况
#     allocation = [[] for _ in range(num_people)]
#     # 对每个人的奖励按降序排列球的索引
#     sorted_indices = np.argsort(-single_ru_mapper, axis=1)
    
#     # 首先分配每个人至少一个球，确保每个人都有球
#     for user in range(num_people):
#         for RU_index in sorted_indices[user]:
#             if not allocated[RU_index]:
#                 allocation[user].append(RU_index)
#                 allocated[RU_index] = True
#                 RUs_per_user[user] += 1
#                 break  # 继续下一个人的分配
    
#     # 如果有剩余的球，继续分配直到所有球都被分配
#     while not all(allocated):
#         for user in range(num_people):
#             for RU_index in sorted_indices[user]:
#                 if not allocated[RU_index] and RUs_per_user[user] < 3:
#                     allocation[user].append(RU_index)
#                     allocated[RU_index] = True
#                     RUs_per_user[user] += 1
#                     if RUs_per_user[user] == 3:
#                         break  # 当前人已分配满，继续下一个人
#                 if all(allocated):  # 如果所有的球都分配了，结束循环
#                     break
#             if all(allocated):  # 再次检查，如果所有球都分配完毕，则结束外层循环
#                 break

#     # 结果矩阵，记录分配情况
#     result_matrix = np.zeros((num_people, num_RUs), dtype=int)
#     for user, RUs in enumerate(allocation):
#         for RU in RUs:
#             result_matrix[user, RU] = 1

#     return result_matrix

# # 示例奖励矩阵
# single_ru_mapper = np.random.rand(5, 8)

# # 执行分配函数
# result_matrix = allocate_RUs(single_ru_mapper)
# print("奖励矩阵:\n", single_ru_mapper)
# print("分配矩阵:\n", result_matrix)

import numpy as np

def allocate_RUs_no_min(matrix):
    rewards = np.array(matrix)
    m, n = rewards.shape
    RUs_per_user = {i: 0 for i in range(m)}
    allocation = {j: None for j in range(n)}

    # 每个球的分配逻辑
    for RU in range(n):
        # 对于每个球，获取奖励值排序
        sorted_indices = np.argsort(-rewards[:, RU])

        # 分配球直到找到合适的人选
        for user_index in sorted_indices:
            if RUs_per_user[user_index] < 3:
                allocation[RU] = user_index
                RUs_per_user[user_index] += 1
                break

    # 创建一个5×8的表格，初始化为0
    allocation_table = np.zeros((m, n), dtype=int)

    # 填充表格：为每个人拿到的球标记1
    for RU, user in allocation.items():
        if user is not None:  # 确保球被分配了
            allocation_table[user, RU] = 1

    return allocation_table

# 测试函数
matrix = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 3, 4, 5, 6, 7, 8, 1],
    [3, 4, 5, 6, 7, 8, 1, 2],
    [4, 5, 6, 7, 8, 1, 2, 3],
    [5, 6, 7, 8, 1, 2, 3, 4]
]

allocation_table = allocate_RUs(matrix)
print("RUs allocation table:")
print(allocation_table)