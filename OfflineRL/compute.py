#%% Imports
# -------- General -------- #
import os 
from datetime import datetime 
import json

# -------- PyTorch -------- #
import torch
from torch.utils.tensorboard import SummaryWriter

# -------- Scripts -------- #
from OfflineRL.actor_2D import * 
from OfflineRL.InteractiveReplayBuffer import *
from envs.drone_2D import * 
from infrastructure.ReplayBuffer import * 
import infrastructure.utils as utils

#%% Functions for computing the average reward and average distance to target
def compute_optimal_directions_2D(obs, num_agents, num_obstacles, num_targets):
    # Compute the optimal directions for each agent to move towards the target
    # obs.shape: (num_agents+num_obstacles+num_targets)*2 if not batched
    # obs.shape: (batch_size, (num_agents+num_obstacles+num_targets)*2) if batched

    if obs.dim() == 1: # for non-batched observations
        # Isolate the agent, obstable, and target positions from the observation
        agents = obs[0:num_agents*2].reshape(num_agents,2)
        obstacles = obs[num_agents*2:num_agents*2+num_obstacles*2].reshape(num_obstacles,2)
        targets = obs[num_agents*2+num_obstacles*2:].reshape(num_targets,2)
        # Compute the distance between the agents and the obstacles as well as the agents and the targets
        aoDist = torch.cdist(agents, obstacles, p=2)
        atDist = torch.cdist(agents, targets, p=2)
        # Compute the minimum distance between the agents and the obstacles as well as the agents and the targets
        min_aoDist, min_aoDist_ind = torch.min(aoDist, dim=1)
        min_atDist, min_atDist_ind = torch.min(atDist, dim=1)
        # Compute the direction of the minimum distance between the agents and the obstacles as well as the agents and the targets
        aoDir = F.normalize(obstacles[min_aoDist_ind] - agents, dim=1)
        atDir = F.normalize(targets[min_atDist_ind] - agents, dim=1)
        # Weight the direction of the minimum distance between the agents and the obstacles and the agents and the targets
        weights_ao = 1.0/(min_aoDist+1e-6)
        weights_at = 5.0/(min_atDist+1e-6)
        # Compute the weighted average of the direction of the minimum distance between the agents and the obstacles and the agents and the targets
        optimal_directions = weights_at.reshape(-1,1)*atDir # - weights_ao.reshape(-1,1)*aoDir
        # Normalize the optimal directions
        optimal_directions = F.normalize(optimal_directions, dim=1)
        # Flatten the optimal directions
        optimal_directions = optimal_directions.reshape(-1)

    elif obs.dim() == 2: # for batched observations
        # Isolate the agent, obstable, and target positions from the observation
        batch_size = obs.shape[0]
        agents = obs[:,0:num_agents*2].reshape(batch_size,num_agents,2)
        obstacles = obs[:,num_agents*2:num_agents*2+num_obstacles*2].reshape(batch_size,num_obstacles,2)
        targets = obs[:,num_agents*2+num_obstacles*2:].reshape(batch_size,num_targets,2)
        # Compute the distance between the agents and the obstacles as well as the agents and the targets
        aoDist = torch.cdist(agents, obstacles, p=2)
        atDist = torch.cdist(agents, targets, p=2)
        # Compute the minimum distance between the agents and the obstacles as well as the agents and the targets
        min_aoDist, min_aoDist_ind = torch.min(aoDist, dim=2)
        min_atDist, min_atDist_ind = torch.min(atDist, dim=2)
        # Compute the direction of the minimum distance between the agents and the obstacles as well as the agents and the targets
        aoDir = F.normalize(obstacles[torch.arange(batch_size).reshape(-1,1),min_aoDist_ind] - agents, dim=2)
        atDir = F.normalize(targets[torch.arange(batch_size).reshape(-1,1),min_atDist_ind] - agents, dim=2)
        # Weight the direction of the minimum distance between the agents and the obstacles and the agents and the targets
        weights_ao = 1.0/(min_aoDist+1e-6)
        weights_at = 5.0/(min_atDist+1e-6)
        # Compute the weighted average of the direction of the minimum distance between the agents and the obstacles and the agents and the targets
        optimal_directions = weights_at.reshape(batch_size,-1,1)*atDir # - weights_ao.reshape(batch_size,-1,1)*aoDir
        # Normalize the optimal directions
        optimal_directions = F.normalize(optimal_directions, dim=2)
        # Flatten the optimal directions
        optimal_directions = optimal_directions.reshape(batch_size,-1)

    else:
        raise ValueError("The observation vector should be a 1D or 2D tensor")
    
    return optimal_directions