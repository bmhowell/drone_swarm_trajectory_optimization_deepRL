import torch 
import torch.nn.functional as F
import numpy as np 

class OUNoise(object):
    def __init__(self, act_size, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = act_size
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        pt_ou_state = from_numpy(ou_state)
        noisy_action = action + pt_ou_state
        return pt_normalize_actions(noisy_action)

class OUNoise_2D(object):
    def __init__(self, act_size, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = act_size
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        pt_ou_state = from_numpy(ou_state)
        noisy_action = action + pt_ou_state
        return pt_normalize_actions_2D(noisy_action)

def from_numpy(mat):
    return torch.from_numpy(mat).float()

def to_numpy(mat):
    return mat.detach().numpy()

def reshape_action(action):
    return action.reshape(action.size/3,3)

def pt_normalize_actions(action):

    if action.dim() == 1: # There is only one dimension if the input is not batched. 
        num_agents = int(len(action)/3)
        action_mat = action.reshape(num_agents, 3)
        action_normalized = F.normalize(action_mat, p=2, dim=1)
        action = action_normalized.reshape(num_agents*3)

    elif action.dim() == 2: # There are two dimensions if the input is batched, in which case action.shape = [batch_size, num_agents*3]
        batch_size = action.shape[0]
        num_agents = int(action.shape[1]/3)

        action_mat = action.reshape(batch_size, num_agents, 3)
        action_normalized = F.normalize(action_mat, p=2, dim=2)
        action = action_normalized.reshape(batch_size, num_agents*3)

    else:
        raise Exception("the shape of your action is larger than 2!")

    return action

def pt_normalize_actions_2D(action):

    if action.dim() == 1: # There is only one dimension if the input is not batched. 
        num_agents = int(len(action)/2)
        action_mat = action.reshape(num_agents, 2)
        action_normalized = F.normalize(action_mat, p=2, dim=1)
        action = action_normalized.reshape(num_agents*2)

    elif action.dim() == 2: # There are two dimensions if the input is batched, in which case action.shape = [batch_size, num_agents*3]
        batch_size = action.shape[0]
        num_agents = int(action.shape[1]/2)

        action_mat = action.reshape(batch_size, num_agents, 2)
        action_normalized = F.normalize(action_mat, p=2, dim=2)
        action = action_normalized.reshape(batch_size, num_agents*2)

    else:
        raise Exception("the shape of your action is larger than 2!")

    return action
