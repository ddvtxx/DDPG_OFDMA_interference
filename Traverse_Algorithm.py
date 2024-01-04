from itertools import permutations
import numpy as np 
import environment_simulation_move as env

def generate_box_distributions(perm, ball_index, current_distribution, boxes, all_distributions):
    # 如果所有球都已分配，并且使用了正确数量的框，则保存分配
    if ball_index == len(perm) and len(current_distribution) == boxes:
        all_distributions.append(current_distribution)
        return

    # 如果当前分配已经使用了所有框，结束当前路径
    if len(current_distribution) == boxes:
        return

    # 尝试将1到3个球分配到当前框
    for i in range(1, 4):
        if ball_index + i <= len(perm):
            # 递归的创建下一个框的分配
            generate_box_distributions(perm, ball_index + i, current_distribution + [perm[ball_index:ball_index + i]], boxes, all_distributions)

def generate_distributions(all_balls, boxes):
    all_distributions = []
    # 生成所有球的全排列
    for perm in permutations(all_balls):
        # 为当前排列生成所有可能的框分配
        generate_box_distributions(perm, 0, [], boxes, all_distributions)
    return all_distributions

# 设置球和框的数量
balls = [1,2,3,4,5,6,7,8]  # 假设有8个球，每个球都是唯一的
boxes = 5  # 5个框

# 获取所有分配方案
all_distributions = generate_distributions(balls, boxes)

# 打印所有可能的分配方案数量（前10个示例）
print(f"总共有 {len(all_distributions)} 种分配方案。")

action_array = []
for distribution in all_distributions:  # 假设我们只打印前10个示例
    action = np.zeros([5,8])
    for i in range(5):
        for j in range(3):
            if j>=len(distribution[i]):
                break
            else:
                action[i][distribution[i][j]-1]=1
    action_array.append(action)
    #print(action)

numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 3

test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
x_init,y_init = test_env.senario_user_local_init()
x,y = x_init,y_init
userinfo = test_env.senario_user_info(x,y)
channel_gain_obs = test_env.channel_gain_calculate()
best_bitrate = 0
best_action = []
index = 0
for i in action_array:
    #for j in action_array:
    #    for m in action_array:
    #        for n in action_array:
    ru_4map = np.vstack((test_env.n_AP_RU_mapper(), i.reshape(1,5,8)))
    tem_bitrate = test_env.calculate_4_cells(ru_4map)
    index+=1
    print('This is ' + str(index) + 'times.')
    if tem_bitrate > best_bitrate:
        best_bitrate = tem_bitrate
        best_action = ru_4map
print("The best action is:")
print(best_action[3])
print("This is the best bitrate")
print(best_bitrate)

