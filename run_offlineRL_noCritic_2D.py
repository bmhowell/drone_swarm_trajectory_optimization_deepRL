#%% Imports
# -------- General -------- #
import os 
from datetime import datetime 
import json

# -------- PyTorch -------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# -------- Scripts -------- #
from OfflineRL.actor_2D import * 
from OfflineRL.InteractiveReplayBuffer import *
from envs.drone_2D import * 
from infrastructure.ReplayBuffer import * 
import infrastructure.utils as utils
from OfflineRL.compute import *

#%% Inputs 
# ----------- NOTE: Change this section to use such that it may be ----------- #
# -----------       run from the command line                      ----------- #

# -------- Training -------- #
num_episodes = 10
num_time_steps_per_episode = 300
num_epochs = 1000
batch_size = 5000
gamma = 0.95
tau   = 0.05
num_actor_gradient_steps = 100
num_critic_gradient_steps = 100

# -------- Environment -------- #
num_agents = 2
num_obstables = 1
num_targets = 2

obs_size = int(num_agents*2 + num_obstables*2 + num_targets*2) # int(num_agents*2*3 + num_agents*2 + num_obstables * 3 + num_targets * 5)
act_size = num_agents*2 # x,y,z directions of the propulsion force for each agent  

# -------- Neural network parameters -------- #
hidden_size = 256
lr_critic = 0.001
lr_actor = 0.001

# -------- ReplayBuffer -------- #
replay_buffer_max_size = 50000 
create_replay_buffer = True
replay_buffer_name = "replay_buffer_2D_{}agent_{}obs_{}target_size{}.pkl".format(num_agents, num_obstables, num_targets, replay_buffer_max_size)
if not os.path.exists("buffers"):
    os.makedirs("buffers")
path2replay_buffer = os.path.join("buffers", replay_buffer_name)
if os.path.exists(path2replay_buffer):
    create_replay_buffer = False

# -------- Noise -------- #
noise_toggle = "on"
if noise_toggle == "on":
    mu = 0.0
    theta = 0.15
    max_sigma = 0.3
    min_sigma = 0.3 
    decay_period = 100000
elif noise_toggle == "off":
    mu = 0.0
    theta = 0.0
    max_sigma = 0.0
    min_sigma = 0.0 
    decay_period = 1

# -------- Logging -------- #
visualizationOneInNrollouts = 1
logdir = 'runs'
exp_name = '2D_offlineRL_gameOfDrones'
now = datetime.now()
savePath = exp_name + '_' + now.strftime("%Y_%m_%d-%H_%M_%S")
tensorboardPath = os.path.join(logdir, savePath)
writer = SummaryWriter(tensorboardPath)

#%% Package all of the inputs into a dictionary and save it to tensorboardPath
config =   {'num_episodes':num_episodes, 
            'num_time_steps_per_episode': num_time_steps_per_episode,
            'batch_size': batch_size,
            'gamma': gamma,
            'tau': tau,
            'num_actor_gradient_steps': num_actor_gradient_steps,
            'num_critic_gradient_steps': num_critic_gradient_steps,
            # 'update_a_every_x_episodes': update_a_every_x_episodes,
            'num_agents': num_agents,
            'num_obstables': num_obstables,
            'num_targets': num_targets,
            'obs_size': obs_size,
            'act_size': act_size,
            'hidden_size': hidden_size,
            'lr_critic': lr_critic,
            'lr_actor': lr_actor,
            'replay_buffer_max_size': replay_buffer_max_size,
            'create_replay_buffer': create_replay_buffer,
            'path2replay_buffer': path2replay_buffer,
            'noise_toggle': noise_toggle,
            'mu': mu,
            'theta': theta,
            'max_sigma': max_sigma,
            'min_sigma': min_sigma,
            'decay_period': decay_period,
            'visualizationOneInNrollouts': visualizationOneInNrollouts,
            'logdir': logdir,
            'exp_name': exp_name,
            'now': now,
            'savePath': savePath,
            'tensorboardPath': tensorboardPath
          } 

# Save hyperparameter configuration
with open(tensorboardPath + '/hyperparameters.pkl', 'wb') as f:
    pickle.dump(config, f)

#%% Initialize the enviroment 
env = GameOfDronesEnv(num_agents, num_obstables, num_targets)

#%% Initialize the actor and critic networks
actor = Actor(obs_size, hidden_size, act_size)
critic = Critic(obs_size + act_size, hidden_size, 1) # 1 output for the Q-value

# Define the loss function and optimizer for the critic 
critic_loss_function = torch.nn.MSELoss()
critic_optimizer     = torch.optim.Adam(critic.parameters(), lr=lr_critic)

# Define the optimizer for the actor 
actor_loss_function  = torch.nn.MSELoss()
actor_optimizer      = torch.optim.Adam(actor.parameters(), lr=lr_actor)

#%% Initialize the Replay Buffer 
ReplayBuffer = Memory(replay_buffer_max_size)

#%% Initialize the noise model
noise_model = utils.OUNoise_2D(act_size, mu, theta, max_sigma, min_sigma, decay_period)

#%% Create or load replay buffer 
if create_replay_buffer:
    # Create the replay buffer
    print("Creating the replay buffer...")
    ReplayBuffer = createReplayBuffer(ReplayBuffer, env, actor, noise_model, writer, config, randomAgentInit=True, randomTargetInit=True)
else: 
    # Load the replay buffer
    print("Loading the replay buffer...")
    ReplayBuffer.loadBuffer(path2replay_buffer)

#%% Perform offline RL on the replay buffer
print("Performing offline RL on the replay buffer...")

# For each epoch
for epoch in range(num_epochs):
    
    print("Epoch: ", epoch)

    # Sample a batch from the ReplayBuffer
    obs_t_B, obs_t_Plus1_B, a_t_B, reward_t_B, done_t_B = ReplayBuffer.sample(batch_size) # All pulled from the ReplayBuffer are numpy arrays

    # NOTE regarding the batching. PyTorch is set up such that the first dimension is the batch dimension.
    # Therefore, if batch_size = 3 and obs_size = 32
    # obs_t_B.size() = torch.size([3, 32])

    # Use a batch of transitions to 1.) update critic and 2.) update actor
    # But first, let's change everything to torch tensors
    obs_t_B = utils.from_numpy(obs_t_B)
    obs_t_Plus1_B = utils.from_numpy(obs_t_Plus1_B)
    a_t_B = utils.from_numpy(a_t_B)
    reward_t_B = utils.from_numpy(reward_t_B)
    done_t_B = utils.from_numpy(done_t_B)

    # ------ Update the actor ------ #
    avg_actor_loss = np.zeros((num_actor_gradient_steps))
    for t in range(num_actor_gradient_steps):
        # Use your latest actor to predict which action to take 
        a_t_B = actor.forward(obs_t_B)
        # Calculate the optimal action using optimal action function 
        opt_dir = compute_optimal_directions_2D(obs_t_B, num_agents, num_obstables, num_targets)
        # Calculate the loss
        assert opt_dir.size() == a_t_B.size()
        actor_loss = actor_loss_function(a_t_B, opt_dir)
        # Zero out the gradients and take a step of gradient descent 
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # Save the loss
        avg_actor_loss[t] = actor_loss

    # Save the actor and critic 
    torch.save(actor.state_dict(), tensorboardPath + '/actor.pth')
    
    # Save the losses to tensorboard
    writer.add_scalar('losses/avg_actor_per_epoch', np.mean(avg_actor_loss), epoch)

#%% Now that the actor and critic have been trained, let's apply them to new environments 
print('Training complete. Now testing the trained actor on new environments')
for episode in range(num_episodes):
    # Reset the environment for each episode
    print("Episode #%d" % episode)
    env.reset(seed=episode, randomAgentInitialization=True, randomTargetInitialization=True) # Or alternatively env.reset(seed=episode)

    for t in range(num_time_steps_per_episode):

        # Find out where you currently are 
        obs_t = env.get_current_observation()
        # Use the actor to predict an action from the current state
        a_t = actor.forward(utils.from_numpy(obs_t)) # the actor's forward pass needs a torch.Tensor
        # Convert action to numpy array 
        a_t = utils.to_numpy(a_t)
        # Take a step in the environment
        obs_t, obs_t_Plus1, reward_t, done_t = env.step(a_t) # the env needs a numpy array

        # Visualize one in every n rollouts
        if episode % visualizationOneInNrollouts == 0:
            if t == 0:
                print("Visualizing episode #%d" % episode)
            visualizationPath = os.path.join(tensorboardPath,'episode{}'.format(episode))
            if not os.path.exists(visualizationPath):
                os.makedirs(visualizationPath)
            env.visualize(savePath=visualizationPath)

        if done_t is True:
            writer.add_scalar('debug/rollout_length', t, episode)
            print('Episode done')
            break

writer.close()
