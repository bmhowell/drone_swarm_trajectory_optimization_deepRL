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
from DDPG.networks import * 
from envs.drone import * 
from infrastructure.ReplayBuffer import * 
import infrastructure.utils as utils

#%% Inputs 
# ----------- NOTE: Change this section to use such that it may be ----------- #
# -----------       run from the command line                      ----------- #

# -------- Training -------- #
num_episodes = 50
num_time_steps_per_episode = 100
batch_size = 2
gamma = 0.95
tau   = 0.05
num_actor_gradient_steps = 10
num_critic_gradient_steps = 10
update_a_and_c_every_x_episodes = 1

# -------- Environment -------- #
num_agents = 1
num_obstables = 0
num_targets = 1

obs_size = int(num_agents*2*3 + num_agents*2 + num_obstables * 3 + num_targets * 5)
act_size = num_agents*3 # x,y,z directions of the propulsion force for each agent  

# -------- Neural network parameters -------- #
hidden_size = 64
lr_critic = 0.01
lr_actor = 0.01

# -------- ReplayBuffer -------- #
replay_buffer_max_size = 1000000

# -------- Noise -------- #
mu = 0.0
theta = 0.15
max_sigma = 0.3
min_sigma = 0.3 
decay_period = 100000

# -------- Logging -------- #
logdir = 'runs'
exp_name = 'gameOfDrones'
now = datetime.now()
savePath = exp_name + '_' + now.strftime("%Y_%m_%d-%H_%M_%S")
tensorboardPath = os.path.join(logdir, savePath)
writer = SummaryWriter(tensorboardPath)

#%% Package all of the inputs into a dictionary and save it to tensorboardPath
config = {'num_episodes':num_episodes, } # TODO: Finish this dictionary or just implement argparse

#%% Initialize the enviroment 
env = GameOfDronesEnv(num_agents, num_obstables, num_targets)

#%% Initialize the actor, critic, and target networks 
actor = Actor(obs_size, hidden_size, act_size)
actor_target = Actor(obs_size, hidden_size, act_size)
critic = Critic(obs_size + act_size, hidden_size, 1) # act_size)
critic_target = Critic(obs_size + act_size, hidden_size, 1) # act_size)

# Initialize the loss functions and the optimizers 
# Define the loss function and optimizer for the critic 
critic_loss_function = torch.nn.MSELoss()
critic_optimizer     = torch.optim.Adam(critic.parameters(), lr=lr_critic)

# Define the optimizer for the actor 
actor_optimizer      = torch.optim.Adam(actor.parameters(), lr=lr_actor)

# We initialize the target networks as copies of the original networks
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

#%% Initialize the Replay Buffer 
ReplayBuffer = Memory(replay_buffer_max_size)

#%% Initialize the noise model
noise_model = utils.OUNoise(act_size, mu, theta, max_sigma, min_sigma, decay_period)

#%% Main training loop 

# Allocate memory for saving variables 
avg_critic_loss     = np.zeros((num_episodes))
avg_actor_loss      = np.zeros((num_episodes))
avg_episode_reward  = np.zeros((num_episodes))

reward_dict = {}
reward_dict_keys = []
for episode in range(num_episodes):
    reward_dict_keys.append('episode{}'.format(episode))

num_env_step = 0
final_env_stepper = 0

for episode in range(num_episodes):

    # Resent the environment for each episode
    print("Episode #%d" % episode)
    env.reset(seed=episode) # Or alternatively env.reset(seed=episode)

    # Allocate memory for saving variables throughout each episode
    critic_losses = np.zeros((num_time_steps_per_episode))
    actor_losses  = np.zeros((num_time_steps_per_episode))
    rewards       = np.zeros((num_time_steps_per_episode))

    for t in range(num_time_steps_per_episode):

        # Find out where you currently are 
        obs_t = env.get_current_observation()
        
        # Use the actor to predict an action from the current state
        a_t = actor.forward(utils.from_numpy(obs_t)) # the actor's forward pass needs a torch.Tensor
        a_t = noise_model.get_action(a_t, num_env_step)
        # Convert action to numpy array 
        a_t = utils.to_numpy(a_t)
        # a_t = test_action.flatten()
        # a_t = 2*np.random.random(num_agents*3)-1
        obs_t, obs_t_Plus1, reward_t, done_t = env.step(a_t) # the env needs a numpy array
        if t == 0: 
            writer.add_scalar('rewards/rewards_all_episodes',t, num_env_step)
        else:
            writer.add_scalar('rewards/rewards_all_episodes',reward_t, num_env_step)
        num_env_step += 1

        if episode == num_episodes - 1:
            env.visualize(savePath=tensorboardPath)
            writer.add_scalar('rewards/final_episode_reward',reward_t, final_env_stepper)
            final_env_stepper += 1
            
        ReplayBuffer.push(obs_t, obs_t_Plus1, a_t, reward_t, done_t) # All pushed into the ReplayBuffer need to be numpy arrays
        writer.add_scalar('debug/ReplayBufferSize', len(ReplayBuffer), num_env_step)

        if len(ReplayBuffer) > batch_size and episode % update_a_and_c_every_x_episodes == 0: # As soon as the ReplayBuffer has accumulated enough memory, perform RL

            # Sample a batch from the ReplayBuffer
            obs_t_B, obs_t_Plus1_B, a_t_B, reward_t_B, done_t_B = ReplayBuffer.sample(batch_size) # All pulled from the ReplayBuffer are numpy arrays

            # Note regarding the batching. PyTorch is set up such that the first dimension is the batch dimension.
            # Therefore, if batch_size = 3 and obs_size = 32
            # obs_t_B.size() = torch.size([3, 32])

            # Use a batch of transitions to 1.) update critic and 2.) update actor
            # But first, let's change everything to torch tensors
            obs_t_B = utils.from_numpy(obs_t_B)
            obs_t_Plus1_B = utils.from_numpy(obs_t_Plus1_B)
            a_t_B = utils.from_numpy(a_t_B)
            reward_t_B = utils.from_numpy(reward_t_B)
            done_t_B = utils.from_numpy(done_t_B)

            # ------ 1.) Update the critic ------ #
            for _ in range(num_critic_gradient_steps):
                # Get current Q-estimates
                Q_t_B = critic.forward(obs_t_B,a_t_B)
                # Use actor to predict next action given next states
                a_t_plus1_B = actor_target.forward(obs_t_Plus1_B)
                # Define the target Q's given the reward and the discounted next Q's
                target_Qs = reward_t_B + gamma * critic_target.forward(obs_t_Plus1_B, a_t_plus1_B) * (1-done_t_B)
                # NOTE: Regarding the line above. There is a predicted Q value for every action. But there is only one reward for each group of actions. 
                # Assert that shapes of the estimated Q's and the target Q's are the same 
                assert Q_t_B.size() == target_Qs.size()
                # Feed the estimate Q values and target Q values
                critic_loss = critic_loss_function(Q_t_B, target_Qs)
                # Zero out the gradients and take a step of gradient descent
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            # ------ 2.) Update the actor ------ #
            for _ in range(num_actor_gradient_steps):
                # Use your latest actor to predict which action to take 
                a_t_B = actor.forward(obs_t_B)
                # Get current Q-estimates
                Q_t_B = critic.forward(obs_t_B,a_t_B)
                # For the actor, we simply wish to maximize the average Q
                # Therefore, we can define our actor loss function as 
                actor_loss = -1 * torch.mean(Q_t_B)
                # Zero out the gradients and take a step of gradient descent 
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            # ------ Finally, we can perform a soft update on the target networks ------ #
            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
        
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

            # ------ Save the rewards, the critic losses, and the actor losses ------ #
            critic_losses[t] = critic_loss
            actor_losses[t]  = actor_loss
            rewards[t]       = reward_t

        # Save variables to tensorboard
        writer.add_scalar('debug/noise_decay',(num_env_step/decay_period), num_env_step)

        if done_t is True:
            writer.add_scalar('debug/rollout_length', t, episode)
            print('Episode done')
            break
    
    if len(ReplayBuffer) > batch_size and episode % update_a_and_c_every_x_episodes == 0:
        writer.add_scalar('losses/avg_critic_loss_per_episode', np.mean(critic_losses), episode)
        writer.add_scalar('losses/avg_actor_loss_per_episode', np.mean(actor_losses), episode)
        writer.add_scalar('rewards/avg_reward_per_episode', np.mean(rewards), episode)

writer.close()
