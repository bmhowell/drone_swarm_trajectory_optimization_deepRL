import torch 
import numpy as np 

def from_numpy(mat):
    return torch.from_numpy(mat).float()

def to_numpy(mat):
    return mat.detach().numpy()

def reshape_action(action):
    return action.reshape(action.size/3,3)
