#%% Imports
# -------- General -------- #

# -------- PyTorch -------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Scripts -------- #
from networks import * 
from drone import * 

#%% Inputs 
# ----------- NOTE: Change this section to use such that it may be ----------- #
# -----------       run from the command line                      ----------- #

# -------- Environment -------- #
num_agents = 2
num_obstables = 4
num_targets = 2 

env = GameOfDronesEnv(num_agents, num_obstables, num_classes)
env.reset()
env.visualize()

# -------- Neural network parameters -------- #
hidden_size = 256
lr = 0.001

#%% Initialize the enviroment 
obs_size = (num_agents * 3 +                # x,y,z coordinates of the agents 
            num_obstables * 3 +             # x,y,z coordinates of the obstacles  
            num_targets * 3 +               # x,y,z coordinates of the targets
            num_classes * (num_agents +     # 
                           num_obstables +  # A one-hot vector for every agent, obstacle, and target
                           num_targets )    # 
            )

act_size = num_agents * 3                   # x,y,z directions of the propulsion force for each agent  

#%% Initialize the actor, critic, and target networks 
actor = Actor(obs_size, hidden_size, act_size)
actor_target = Actor(obs_size, hidden_size, act_size)
critic = Critic(obs_size + act_size, hidden_size, act_size)
critic_target = Critic(obs_size + act_size, hidden_size, act_size)

# We initialize the target networks as copies of the original networks
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)


#%% Main training loop 

# for epsisode in range(num_episodes):

#     for t in range(num_time_steps_per_episode):



# import sys
# import gym
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from ddpg import DDPGagent
# from utils import *
# from ReplayBuffer import * 

# env = NormalizedEnv(gym.make("Pendulum-v1"))

# agent = DDPGagent(env)
# noise = OUNoise(env.action_space)
# batch_size = 128
# rewards = []
# avg_rewards = []

# for episode in range(50):
#     state = env.reset()
#     noise.reset()
#     episode_reward = 0
    
#     for step in range(500):
#         action = agent.get_action(state)
#         action = noise.get_action(action, step)
#         new_state, reward, done, _ = env.step(action) 
#         agent.memory.push(state, action, reward, new_state, done)
        
#         if len(agent.memory) > batch_size:
#             agent.update(batch_size)        
        
#         state = new_state
#         episode_reward += reward

#         if done:
#             sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
#             break

#     rewards.append(episode_reward)
#     avg_rewards.append(np.mean(rewards[-10:]))

# plt.plot(rewards)
# plt.plot(avg_rewards)
# plt.plot()
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()