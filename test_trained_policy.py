#%% Imports
# -------- General -------- #
import os 
from datetime import datetime 

# -------- PyTorch -------- #
import torch
from torch.utils.tensorboard import SummaryWriter

# -------- Scripts -------- #
from OfflineRL.actor_2D import * 
from OfflineRL.InteractiveReplayBuffer import *
from envs.drone_2D import * 
from infrastructure.ReplayBuffer import * 
import infrastructure.utils as utils

#%% Inputs 
# ----------- NOTE: Change this section to use such that it may be ----------- #
# -----------       run from the command line                      ----------- #

# -------- Testing -------- #
num_episodes = 2
num_time_steps_per_episode = 300

# -------- Environment -------- #
num_agents = 2
num_obstables = 0
num_targets = 1

obs_size = int(num_agents*2*3 + num_obstables * 3 + num_targets * 4)
act_size = num_agents*2 # x,y,z directions of the propulsion force for each agent  

# -------- Neural network parameters -------- #
path2actor = "/Users/rdhuff/Desktop/drone_swarm_trajectory_optimization_deepRL/runs/2D_gameOfDrones_2022_12_17-22_25_42/actor.pth"
hidden_size = 256

# -------- Logging -------- #
visualizationOneInNrollouts = 1
logdir = 'runs'
exp_name = '2D_testing_DDPG'
now = datetime.now()
savePath = exp_name + '_' + now.strftime("%Y_%m_%d-%H_%M_%S")
tensorboardPath = os.path.join(logdir, savePath)
writer = SummaryWriter(tensorboardPath)

#%% Initialize the enviroment 
env = GameOfDronesEnv(num_agents, num_obstables, num_targets)

#%% Initialize the actor and critic networks
actor = Actor(obs_size, hidden_size, act_size)

#%% Load the trained policy
actor.load_state_dict(torch.load(path2actor))

#%% Now that the actor and critic have been trained, let's apply them to new environments 
print('Training complete. Now testing the trained actor and critic on new environments')
for episode in range(num_episodes):
    
    # Reset the environment for each episode
    print("Episode #%d" % episode)
    env.reset(seed=episode) # Or alternatively env.reset(seed=episode)

    # Allocate space for the average reward
    avg_reward = 0

    for t in range(num_time_steps_per_episode):

        # Find out where you currently are 
        obs_t = env.get_current_observation()
        # Use the actor to predict an action from the current state
        a_t = actor.forward(utils.from_numpy(obs_t)) # the actor's forward pass needs a torch.Tensor
        # Convert action to numpy array 
        a_t = utils.to_numpy(a_t)
        # Take a step in the environment
        obs_t, obs_t_Plus1, reward_t, done_t = env.step(a_t) # the env needs a numpy array

        # Update the average reward
        avg_reward += reward_t

        # Visualize one in every n rollouts
        if episode % visualizationOneInNrollouts == 0:
            if t == 0:
                print("Visualizing episode #%d" % episode)
            visualizationPath = os.path.join(tensorboardPath,'episode{}'.format(episode))
            if not os.path.exists(visualizationPath):
                os.makedirs(visualizationPath)
            env.visualize(savePath=visualizationPath)

        if done_t is True:
            # Determine if the done is from a collision or a target capture
            if env.target_found is False:
                print('Collision')
                writer.add_scalar('results/collision', 1, episode)
                writer.add_scalar('results/target_captured', 0, episode)
            else:
                print('Target captured')
                writer.add_scalar('results/collision', 0, episode)
                writer.add_scalar('results/target_captured', 1, episode)

            # Log the average reward
            avg_reward = avg_reward / t
            writer.add_scalar('results/avg_reward', avg_reward, episode)

            writer.add_scalar('results/rollout_length', t, episode)
            print('Episode done')
            break