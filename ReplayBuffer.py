import numpy as np
import os

class ReplayBuffer:
  def __init__(self,max_size,numSenario,numAPuser,numRU,batch_size):
    self.memory_size = max_size
    self.batch_size = batch_size
    self.memory_cntr = 0
    self.state_memory = np.zeros((self.memory_size,numSenario,numAPuser,numRU))
    self.next_state_memory = np.zeros((self.memory_size,numSenario,numAPuser,numRU))
    self.action_memory = np.zeros((self.memory_size,numAPuser,numRU))
    self.reward_memory = np.zeros(self.memory_size)
    #self.terminal_memory = np.zeros(self.memory_size,dtype=np.bool)
    self.terminal_memory = np.zeros(self.memory_size,dtype=np.bool_)

  def store_transition(self, state, action, reward, next_state, done):
    index = self.memory_cntr%self.memory_size
    self.state_memory[index,:,:,:] = state
    self.next_state_memory[index,:,:,:] = next_state
    self.action_memory[index,:,:] = action
    self.reward_memory[index] = reward
    self.terminal_memory[index] = done
    self.memory_cntr += 1

  def sample_buffer(self):
    max_memory = min(self.memory_cntr,self.memory_size)
    batch = np.random.choice(max_memory,self.batch_size,replace=False)

    states = self.state_memory[batch,:,:,:]
    states_ = self.next_state_memory[batch,:,:,:]
    actions = self.action_memory[batch,:,:]
    rewards = self.reward_memory[batch]
    dones = self.terminal_memory[batch]

    return states, actions, rewards, states_, dones
  def ready(self):
    return self.memory_cntr >= self.batch_size
  
  

def create_directory(path: str, sub_paths: list):
    for sub_path in sub_paths:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Create path: {} successfully'.format(path+sub_path))
        else:
            print('Path: {} is already existence'.format(path+sub_path))