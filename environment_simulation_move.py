import random
import numpy as np
import math


class environment_base:
    def __init__(self,numUserAP,numRU,Linkmode,RU_mode):
        self.numAP = 4 
        self.numUserAP = numUserAP 
        self.cellShape = 'square'
        self.size = 50 
        self.Linkmode = Linkmode 
        self.heightAP = 2.5 
        self.heightuser = 1.5 
        self.fc = 5
        self.numRU = numRU 
        self.bwRU = 78000 * 26 
        self.N0 = 10 ** (-13.62)
        self.AP_Gr = 12      
        self.user_Gt_list = 10 
        self.RU_mode = RU_mode 
        self.A = 18.7
        self.B = 46.8
        self.C = 20
        self.X = 0
        self.sigma = 3  
        return None

    def senario_user_local_init(self):

        random.seed(9)
        np.random.seed(9)

        # Initialization of arrays to hold user locations
        user_local_x_init = np.zeros((self.numAP,self.numUserAP))
        user_local_y_init = np.zeros((self.numAP,self.numUserAP))        
        for i in range(int(pow(self.numAP,0.5))): 
            for j in range(int(pow(self.numAP,0.5))):
                if self.numUserAP == 5:
                    x = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(self.numUserAP))))
                    y = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(self.numUserAP))))   
                elif self.numUserAP == 6:
                    x = np.zeros(6)
                    y = np.zeros(6)
                    x_p = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(5))))
                    y_p = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(5))))

                    x1 = np.array(list(map(lambda x:np.random.uniform(50*i, 50*(i+1)),range(1))))
                    y1 = np.array(list(map(lambda x:np.random.uniform(50*j, 50*(j+1)),range(1))))
                    x[0:5] = x_p
                    x[-1] = x1
                    y[0:5] = y_p
                    y[-1] = y1
                elif self.numUserAP == 7:
                    x = np.zeros(7)
                    y = np.zeros(7)
                    x_p = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(5))))
                    y_p = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(5))))

                    x1 = np.array(list(map(lambda x:np.random.uniform(50*i, 50*(i+1)),range(1))))
                    y1 = np.array(list(map(lambda x:np.random.uniform(50*j, 50*(j+1)),range(1))))
                    x2 = min(x1) + 5
                    y2 = min(y1) + 5
                    x[0:5] = x_p
                    x[-2] = x1
                    x[-1] = x2
                    y[0:5] = y_p
                    y[-2] = y1
                    y[-1] = y2          
                elif  self.numUserAP == 8: 
                    x = np.zeros(8)
                    y = np.zeros(8)
                    x_p = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(5))))
                    y_p = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(5))))

                    x1 = np.array(list(map(lambda x:np.random.uniform(50*i, 50*(i+1)),range(1))))
                    y1 = np.array(list(map(lambda x:np.random.uniform(50*j, 50*(j+1)),range(1))))    
                    x2 = min(x1) + 5
                    y2 = min(y1) + 5
                    x[0:5] = x_p
                    x[-3] = x1
                    x[-2] = x2
                    x[-1] = x2 + 2
                    y[0:5] = y_p
                    y[-3] = y1
                    y[-2] = y2  
                    y[-1] = y2 + 2     
                elif  self.numUserAP == 9: 
                    x = np.zeros(9)
                    y = np.zeros(9)
                    x_p = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(5))))
                    y_p = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(5))))

                    x1 = np.array(list(map(lambda x:np.random.uniform(50*i, 50*(i+1)),range(1))))
                    y1 = np.array(list(map(lambda x:np.random.uniform(50*j, 50*(j+1)),range(1))))    
                    x2 = min(x1) + 5
                    y2 = min(y1) + 5
                    x[0:5] = x_p
                    x[-4] = x1
                    x[-3] = x2
                    x[-2] = x2 + 2
                    x[-1] = x2 + 8
                    y[0:5] = y_p
                    y[-4] = y1
                    y[-3] = y2  
                    y[-2] = y2 + 2   
                    y[-1] = y2 + 8     
                elif  self.numUserAP == 10: 
                    x = np.zeros(10)
                    y = np.zeros(10)
                    x_p = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(5))))
                    y_p = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(5))))

                    x1 = np.array(list(map(lambda x:np.random.uniform(50*i, 50*(i+1)),range(1))))
                    y1 = np.array(list(map(lambda x:np.random.uniform(50*j, 50*(j+1)),range(1))))    
                    x2 = min(x1) + 5
                    y2 = min(y1) + 5
                    x[0:5] = x_p
                    x[-5] = x1
                    x[-4] = x2
                    x[-3] = x2 + 2
                    x[-2] = x2 + 8
                    x[-1] = x2 + 14
                    y[0:5] = y_p
                    y[-5] = y1
                    y[-4] = y2  
                    y[-3] = y2 + 2   
                    y[-2] = y2 + 8 
                    y[-1] = y2 + 14
                elif  self.numUserAP < 5:
                    x = np.array(list(map(lambda x:random.uniform(50*i, 50*(i+1)),range(5))))
                    y = np.array(list(map(lambda x:random.uniform(50*j, 50*(j+1)),range(5)))) 
                    temp = 5 - self.numUserAP
                    for times in range(temp):
                        x = np.delete(x, [-1], axis=0)             
                        y = np.delete(y, [-1], axis=0)
                user_local_x_init[i*2+j,] = x        
                user_local_y_init[i*2+j,] = y
        if  self.numUserAP > 4: 
            user_local_x_init[0,4] += 5   
            user_local_x_init[1,0] += 5        
        return user_local_x_init,user_local_y_init
    #user_local_x_init = [
    #[x11, x12],  # Coordinates for users at AP1
    #[x21, x22],  # Coordinates for users at AP2
    #[x31, x32],  # Coordinates for users at AP3
    #[x41, x42],  # Coordinates for users at AP4
    #[x51, x52],  # Coordinates for users at AP5
    #[x61, x62],  # Coordinates for users at AP6
    #[x71, x72],  # Coordinates for users at AP7
    #[x81, x82],  # Coordinates for users at AP8
    #[x91, x92]   # Coordinates for users at AP9
#]
    #user_local_y_init = [
    #[y11, y12],  # Coordinates for users at AP1
    #[y21, y22],  # Coordinates for users at AP2
    #[y31, y32],  # Coordinates for users at AP3
    #[y41, y42],  # Coordinates for users at AP4
    #[y51, y52],  # Coordinates for users at AP5
    #[y61, y62],  # Coordinates for users at AP6
    #[y71, y72],  # Coordinates for users at AP7
    #[y81, y82],  # Coordinates for users at AP8
    #[y91, y92]   # Coordinates for users at AP9
#]

    def senario_user_local_move(self,x,y):
        user_local_x_ = np.zeros((self.numAP,self.numUserAP))
        user_local_y_ = np.zeros((self.numAP,self.numUserAP))
        for n in range(self.numAP):
            user_local_x_[n,] =  np.array(list(map(lambda x: x + random.uniform(-0.1,0.1),x[n,]))) 
            user_local_y_[n,] =  np.array(list(map(lambda x: x + random.uniform(-0.1,0.1),y[n,]))) 
        user_local_x =  user_local_x_
        user_local_y =  user_local_y_
        return user_local_x,user_local_y

    def senario_user_info(self,user_local_x,user_local_y):
        self.Userinfo = np.zeros((self.numAP,self.numUserAP,self.numAP))
        for i in range(int(pow(self.numAP,0.5))): 
            for j in range(int(pow(self.numAP,0.5))):
                for k in range(int(pow(self.numAP,0.5))):
                    for m in range(int(pow(self.numAP,0.5))):
                        center = [50 * k + 50 / 2, 50 * m + 50 / 2]                        
                        dis_user2AP = np.sqrt(np.square(user_local_x[i*2+j,] - center[0]) + np.square(user_local_y[i*2+j,] - center[1]))
                        self.Userinfo[i*int(pow(self.numAP,0.5))+j,:,k*int(pow(self.numAP,0.5))+m] = dis_user2AP.T #[维，行，列]
        return self.Userinfo
    
    def cal_user_dis(self, x_coords, y_coords):
        self.user_dis = np.zeros((self.numAP,self.numUserAP))
        for i_ap in range(self.numAP):
            for i_user in range(self.numUserAP):
                for i_ap_c in range(self.numAP):
                    for i_user_c in range(self.numUserAP):
                        if i_ap != i_ap_c and i_user != i_user_c:
                            distance = math.sqrt(
                                (x_coords[i_ap, i_user]-x_coords[i_ap_c, i_user_c])**2 +
                                (y_coords[i_ap, i_user]-y_coords[i_ap_c, i_user_c])**2
                            )
                            self.user_dis[i_ap, i_user] += distance

        return self.user_dis

    def channel_gain_calculate(self):
        #path loss
        Lp = self.A * np.log10(self.Userinfo) + self.B + self.C * np.log10(self.fc / 5.0) + self.X 
        #Shadowing component
        Ls = self.sigma*random.gauss(0,1)       
        #channel power gain
        h = self.user_Gt_list + self.AP_Gr - Lp - Ls   
        # Initialize an array to store the channel gains for different APs, users, 
        #and resource units (RUs).
        self.channel_gain = np.zeros((self.numAP,self.numAP,self.numUserAP,self.numRU))
        # Loop over all APs to calculate the channel gains.
        for i in range(self.numAP):
            for j in range(self.numAP):
                # Tile the channel gains for all resource units.
                h2AP = np.tile(h[j, :, i], (self.numRU, 1)).T
                # Fast fading component (Lf) modeled as an exponential random variable.
                Lf = np.random.exponential(1, h2AP.shape) 
                 # Combine all effects and convert to linear scale by using 10^(x/10).
                h2AP = np.power(10, (h2AP - Lf) / 10)
                # Store the calculated channel gains in the array.
                self.channel_gain[i][j]=h2AP      
        #(ap,ap,user,ru)
        return self.channel_gain
    

    def n_AP_RU_mapper(self):
        # Mode 1: Each user is assigned to a distinct RU without overlap.
        if self.RU_mode == 1:
            # Initialize the mapper with zeros.
            self.ru_mapper = np.zeros((1, self.numUserAP, self.numRU))
            temp = np.zeros((self.numUserAP, self.numRU))
            # Randomly select non-repeating RUs for the users.
            idy = np.random.choice([i for i in range(0, self.numRU)], self.numUserAP, replace=False)
            for idx in range(0, self.numUserAP):
                temp[idx, idy[idx]] = 1
            self.ru_mapper = temp

        # Mode 2: Each user may be mapped to multiple RUs.
        elif self.RU_mode == 2:
            # Initialize variables to keep track of RU allocation.
            ru_num_sum = list(range(self.numRU))
            ru_num_mapper = np.zeros((self.numUserAP, 3)) - 1 
            self.ru_mapper = np.zeros((self.numUserAP, self.numRU))
            
            # Randomly select non-repeating RUs for the first set of users.
            ru_num_frist = np.random.choice(ru_num_sum, self.numUserAP, replace = False) 
            ru_num_unuse = np.delete(ru_num_sum, ru_num_frist)
            # Calculate how many RUs are left unassigned and fill them with -1.
            num_0 = self.numUserAP * 2 - len(ru_num_unuse)
            no_ru = [-1]
            ru_num_unuse = np.append(ru_num_unuse,no_ru*num_0) 
            np.random.shuffle(ru_num_unuse)
            ru_num_unuse = ru_num_unuse.reshape(self.numUserAP,2) 
            # Map users to the random RUs
            for i in range(self.numUserAP):
                ru_num_mapper[i,0] = ru_num_frist[i]
            ru_num_mapper[:,[1]] = ru_num_unuse[:,[0]] 
            ru_num_mapper[:,[2]] = ru_num_unuse[:,[1]]

            # Update the mapper with the assigned RUs.
            for i in range(self.numUserAP):   
                for k in range(3):
                    if ru_num_mapper[i,k] > -1:
                        ru = int(ru_num_mapper[i,k])
                        self.ru_mapper[i,ru] = 1

        # Mode 3: It seems to be a more complex RU allocation based on channel gain.
        elif self.RU_mode == 3:
            #self.numRU from 8 to 80
            # Initialize variables to store channel gains and RU allocations.
            AP_user_channel_gain = np.zeros((3,self.numUserAP,self.numRU))
            ru_3AP = np.zeros((3,self.numUserAP,self.numRU))
            # Get the channel gain between the same APs.
            for i in range(3):
                for j in range(3):
                    if i == j:
                        AP_user_channel_gain[i,:,:] = self.channel_gain[i][j] 
            
            for k in range(3):
                user_list = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,0,0,0]
                key = AP_user_channel_gain[k,:,:]
                
                #change 8 to self.numRU
                for i in range(self.numRU): 
                    max_key = np.argmax(key[:,i])
                    if self.numUserAP <=2:
                        if max_key in user_list :
                            ru_3AP[k,max_key,i] = 1
                            user_list.remove(max_key) 
                        
                    if self.numUserAP >2:
                        if max_key in user_list :
                            ru_3AP[k,max_key,i] = 1
                            user_list.remove(max_key) 
                        else:
                            key[max_key,:] = 0
                            max_key = np.argmax(key[:,i])
                            if max_key not in user_list:
                                key[max_key,:] = 0
                                max_key = np.argmax(key[:,i])
                            
                            user_list.remove(max_key)
                            ru_3AP[k,max_key,i] = 1
            self.ru_mapper = ru_3AP
        return self.ru_mapper

    def water_filling(self, channel_gains, P_total, epsilon=1e-5, max_iterations=1000):
        """
        Perform water filling algorithm on a given set of channel gains for multiple users in multiple scenarios.
        
        Args:
        - channel_gains: 3D numpy array of channel gains for each scenario, user, and sub-channel
        - P_total: total power budget for each user
        - epsilon: tolerance for convergence
        - max_iterations: maximum number of iterations to prevent infinite loop
        
        Returns:
        - power_allocation: 3D numpy array of power allocated to each scenario, user, and sub-channel
        """
        scenarios, users, sub_channels = channel_gains.shape
        power_allocation = np.zeros((scenarios, users, sub_channels))
        channel_gains_total = channel_gains.sum(axis=2)

        # Perform water filling for each user in each scenario
        for scenario in range(scenarios):
            for user in range(users):
                user_channel_gains = channel_gains[scenario, user, :]
                non_zero_gains_indices = user_channel_gains > 0
                non_zero_gains = user_channel_gains[non_zero_gains_indices]
                non_zero_gains = non_zero_gains/channel_gains_total[scenario][user]*10
                num_non_zero_gains = len(non_zero_gains)

                if num_non_zero_gains == 0:
                    continue  # Skip if user has no channel gains

                # Initialize water level based on total power and number of non-zero gains
                water_level = P_total / num_non_zero_gains + np.sum(1 / non_zero_gains) / num_non_zero_gains
                
                iteration = 0
                while iteration < max_iterations:
                    iteration += 1
                    
                    # Calculate power allocation for non-zero gains
                    power_allocation_temp = np.maximum(water_level - 1 / non_zero_gains, 0)
                    
                    # Sum of power allocated must not exceed P_total
                    P_total_used = np.sum(power_allocation_temp)
                    
                    # Check if the total power used is within tolerance
                    if np.abs(P_total - P_total_used) < epsilon:
                        break
                    
                    # Adjust the water level
                    water_level += (P_total - P_total_used) / num_non_zero_gains

                # Ensure we did not exceed maximum iterations
                if iteration == max_iterations:
                    print(f"Warning: Maximum iterations reached for user {user} in scenario {scenario}.")

                # Map allocated power back to original channels, including zeros
                allocated_power = np.zeros_like(user_channel_gains)
                allocated_power[non_zero_gains_indices] = power_allocation_temp
                power_allocation[scenario, user, :] = allocated_power

        return power_allocation

    #calculate the system bit rate
    def calculate_4_cells(self,ru_mapper_nAP):

        self.signal_strength = np.array(list(map(lambda x:self.channel_gain[x][x] * ru_mapper_nAP[x],range(self.channel_gain.shape[0]))))
        # #get how many ru do a certain user have
        # ru_per_user = ru_mapper_nAP.sum(axis=2) 
        # ru_per_user = ru_per_user.reshape(self.numAP, self.numUserAP ,1 )
        # ru_per_user = np.tile(ru_per_user, (1,1,self.numRU))
        # ru_per_user_picked = ru_per_user == 0
        # ru_per_user = ru_per_user + ru_per_user_picked.astype(int)
        # #allocate signal power averagely due to 
        # self.signal_strength = self.signal_strength / ru_per_user
        power_allocation = self.water_filling(self.signal_strength, 1)
        self.signal_strength = self.signal_strength*power_allocation
        
        if self.Linkmode == 'uplink':
            sinr_uplink = np.zeros((self.numAP,self.numUserAP,self.numRU))
            self.n_AP_n_user_bitrate = np.zeros((self.numAP,self.numUserAP,self.numRU))
            for i in range(self.numAP):
                interference = np.zeros((self.channel_gain.shape[2:]))
                interference_uplink = np.zeros((self.numAP,self.numUserAP,self.numRU))
                for j in range(self.numAP):
                    if i!=j:
                        interference = (self.channel_gain[i][j]*ru_mapper_nAP[i]*power_allocation[i]).sum(axis=0).reshape(1,self.numRU)
                        interference_uplink[j] = interference.repeat(self.channel_gain.shape[2],axis=0)
                interference_uplink = interference_uplink.sum(axis=0)
                #calculate the SINR
                sinr_uplink[i] = self.signal_strength[i]/(self.N0 + interference_uplink)
                self.n_AP_n_user_bitrate[i] = self.bwRU * np.log2(1 + sinr_uplink[i])
            self.n_AP_bitrate = self.n_AP_n_user_bitrate.sum(axis=2).sum(axis=1)
            self.system_bitrate = self.n_AP_bitrate.sum(axis=0)
                    
        return self.system_bitrate







