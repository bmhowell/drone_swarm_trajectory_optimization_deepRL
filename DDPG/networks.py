import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        batch_size = state.shape[0]
        num_agents = int(self.output_size/3)

        if x.dim() == 1: # There is only one dimension if the input is not batched. 
            pass
        elif x.dim() == 2: # There are two dimensions if the input is batched, in which case x.shape = [batch_size, num_agents*3]
            action_mat = x.reshape(batch_size, num_agents, 3)
            action_normalized = F.normalize(action_mat, p=2, dim=2)
            print('neural network action')
            print(action_normalized)
            print(action_normalized.shape)
            # action *= 200
            x = action_normalized.reshape(batch_size, num_agents*3)

        return x