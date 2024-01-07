import environment_simulation_move as env
import numpy as np
import pandas as pd


# def water_filling(channel_gains, P_total, epsilon=1e-5, max_iterations=1000):
#     """
#     Perform water filling algorithm on a given set of channel gains for multiple users in multiple scenarios.
    
#     Args:
#     - channel_gains: 3D numpy array of channel gains for each scenario, user, and sub-channel
#     - P_total: total power budget for each user
#     - epsilon: tolerance for convergence
#     - max_iterations: maximum number of iterations to prevent infinite loop
    
#     Returns:
#     - power_allocation: 3D numpy array of power allocated to each scenario, user, and sub-channel
#     """
#     scenarios, users, sub_channels = channel_gains.shape
#     power_allocation = np.zeros((scenarios, users, sub_channels))
#     channel_gains_total = channel_gains.sum(axis=2)

#     # Perform water filling for each user in each scenario
#     for scenario in range(scenarios):
#         for user in range(users):
#             user_channel_gains = channel_gains[scenario, user, :]
#             non_zero_gains_indices = user_channel_gains > 0
#             non_zero_gains = user_channel_gains[non_zero_gains_indices]
#             non_zero_gains = non_zero_gains/channel_gains_total[scenario][user]*10
#             num_non_zero_gains = len(non_zero_gains)

#             if num_non_zero_gains == 0:
#                 continue  # Skip if user has no channel gains

#             # Initialize water level based on total power and number of non-zero gains
#             water_level = P_total / num_non_zero_gains + np.sum(1 / non_zero_gains) / num_non_zero_gains
            
#             iteration = 0
#             while iteration < max_iterations:
#                 iteration += 1
                
#                 # Calculate power allocation for non-zero gains
#                 power_allocation_temp = np.maximum(water_level - 1 / non_zero_gains, 0)
                
#                 # Sum of power allocated must not exceed P_total
#                 P_total_used = np.sum(power_allocation_temp)
                
#                 # Check if the total power used is within tolerance
#                 if np.abs(P_total - P_total_used) < epsilon:
#                     break
                
#                 # Adjust the water level
#                 water_level += (P_total - P_total_used) / num_non_zero_gains

#             # Ensure we did not exceed maximum iterations
#             if iteration == max_iterations:
#                 print(f"Warning: Maximum iterations reached for user {user} in scenario {scenario}.")

#             # Map allocated power back to original channels, including zeros
#             allocated_power = np.zeros_like(user_channel_gains)
#             allocated_power[non_zero_gains_indices] = power_allocation_temp
#             power_allocation[scenario, user, :] = allocated_power

#     return power_allocation

# Example usage:
numAPuser = 5
numRU = 8
numSenario = 4
linkmode = 'uplink'
ru_mode = 4
episode = 200
max_iteration = 200
system_bitrate_history = []
system_ave_bitrate_history = []
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)
x,y = test_env.senario_user_local_init()

for i_episode in range(episode):
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
    # print('i_episode =',i_episode, 'system_bitrate =',bitrate)
    # print(x)
bitrate_ave = np.mean(system_bitrate_history)
print(bitrate_ave)

for i in range(600):
    system_ave_bitrate_history.append(bitrate_ave)


dataframe=pd.DataFrame({'bitrate':system_ave_bitrate_history})
dataframe.to_csv("./result/bitrate_water_filling_general_seed_9.csv", index=False,sep=',')


# channel_gains = np.array([
#     # Scenario 1
#     [
#         # User 1
#         [2.57243092e-07, 1.87520480e-07, 1.88574745e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#         # User 2
#         [2.57243092e-07, 1.87520480e-07, 1.88574745e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
#     ],
#     # Scenario 2
#     [
#         # User 1
#         [2.57243092e-07, 1.87520480e-07, 1.88574745e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
#         # User 2
#         [2.57243092e-07, 1.87520480e-07, 1.88574745e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
#     ]
# ])
# P_total = 1  # Total power budget for each user

# power_allocation = water_filling(channel_gains, P_total)

# print("Power Allocation: ", power_allocation)