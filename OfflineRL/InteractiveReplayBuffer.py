#%% Imports
# -------- General -------- #
import os 
from datetime import datetime 

# -------- PyTorch -------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# -------- Scripts -------- #
from DDPG.networks_2D import * 
from envs.drone_2D import * 
from infrastructure.ReplayBuffer import * 
import infrastructure.utils as utils

def createReplayBuffer(ReplayBuffer, env, actor, noise_model, writer, config, randomAgentInit=False, randomTargetInit=True):
    """
    Creates a replay buffer by interacting with the environment.
    """
    # Variables
    replay_buffer_max_size = config['replay_buffer_max_size']
    path2replay_buffer = config['path2replay_buffer']

    # Create the Replay Buffer
    episode = 0
    num_env_step = 0
    while len(ReplayBuffer) < replay_buffer_max_size:

        # Keep track of the episode
        print("Episode #%d" % episode)
        # Reset the environment for each episode
        env.reset(seed=episode, randomAgentInitialization=randomAgentInit, randomTargetInitialization=randomTargetInit) 

        # Find out where you currently are 
        obs_t = env.get_current_observation()

        # Step through the environment until the episode is done
        done_t = False
        while not done_t:
            # Use the actor to predict an action from the current state
            a_t = actor.forward(utils.from_numpy(obs_t)) # the actor's forward pass needs a torch.Tensor
            a_t = noise_model.get_action(a_t, num_env_step) # Add noise to the action
            # Convert action to numpy array 
            a_t = utils.to_numpy(a_t)
            # Take a step in the environment
            obs_t, obs_t_Plus1, reward_t, done_t = env.step(a_t) # the env needs a numpy array
            num_env_step += 1
            # Add the transition to the replay buffer
            ReplayBuffer.push(obs_t, obs_t_Plus1, a_t, reward_t, done_t) # All pushed into the ReplayBuffer need to be numpy arrays
            # Save the ReplayBuffer
            ReplayBuffer.saveBuffer(path2replay_buffer)
            # Keep track of ReplayBuffer size
            writer.add_scalar('debug/ReplayBufferSize', len(ReplayBuffer), num_env_step)
            # If the episode is done, break out of the loop
            if done_t:
                break

        # Increment the episode
        episode += 1

    return ReplayBuffer

