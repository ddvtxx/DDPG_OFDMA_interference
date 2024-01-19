import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# Function for initializing the weights of the network
def weight_init(m):  
    if isinstance(m, nn.Linear):
        # Xavier normal initialization for linear layers
        nn.init.xavier_normal_(m.weight)
        # Zero initialization for biases
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        # Constant initialization for BatchNorm layers
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, 
                 fc1_dim, fc2_dim, fc3_dim, fc4_dim):
        super(ActorNetwork, self).__init__()
        # Flatten the incoming state
        self.input_layer = nn.Flatten() 
        # self.input_layer = nn.Flatten(0,1)
        # Define the fully connected layers with LayerNorm
        self.fc1 = nn.Linear(state_dim, fc1_dim) 
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc4 = nn.Linear(fc3_dim, fc4_dim)

        # The final layer to produce action values
        self.action = nn.Linear(fc4_dim, action_dim)

        # Optimizer for the actor network
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # Apply weight initialization to the network
        self.apply(weight_init)
        # Move the network to the specified device (e.g., GPU)
        self.to(device)

    def forward(self, state):
        # Forward pass through the network with ReLU activations and LayerNorm
        x = T.relu(self.ln1(self.fc1(self.input_layer(state))))
        x = T.relu(self.ln2(self.fc2(x)))
        x = T.relu(self.fc3(x))
        x = T.relu(self.fc4(x))
        
        # Apply Gumbel-Softmax to get a differentiable sample
        action = F.gumbel_softmax(self.action(x), tau=1, hard=False, eps = 1e-5) 
        # Reshape the action to match the input state dimensions
        # action = T.reshape(action,(state.shape[0],state.shape[2],state.shape[3]))
        return action
    
    # Function for saving the network's state
    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    # Function for loading the network's statez
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, 
                 fc1_dim, fc2_dim, fc3_dim, fc4_dim):
        super(CriticNetwork, self).__init__()
        # self.input_layer = nn.Flatten(1,3) 
        self.input_layer = nn.Flatten(1,-1) 
        # self.input_layer = nn.Flatten(0,1)
        self.fc1 = nn.Linear(state_dim*2, fc1_dim) 
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc4 = nn.Linear(fc3_dim, fc4_dim)

       
        self.q = nn.Linear(fc4_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.001)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):  
      
        # action = T.unsqueeze(action,1) 
        # action = action.expand(state.shape)
        state_shape = state.shape
        action = action.reshape((state_shape[0],state_shape[1],state_shape[2],state_shape[3]))
        state_action = T.cat((state,action),2) 
        x = T.relu(self.ln1(self.fc1(self.input_layer(state_action))))
        x = T.relu(self.ln2(self.fc2(x)))
        x = T.relu(self.fc3(x))
        x = T.relu(self.fc4(x))
        

        
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))
