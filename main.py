#%% Imports
# -------- General -------- #

# -------- PyTorch -------- #
from gzip import READ
from modulefinder import ReplacePackage
import re
from readline import replace_history_item
# from selectors import EpollSelector
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Scripts -------- #
from DDPG.networks import * 
from drone import * 
from DDPG.ReplayBuffer import * 
import utils

#%% Inputs 
# ----------- NOTE: Change this section to use such that it may be ----------- #
# -----------       run from the command line                      ----------- #

# -------- Training -------- #
num_episodes = 1000
num_time_steps_per_episode = 40
batch_size = 3
gamma = 0.95
tau   = 0.05

# -------- Environment -------- #
num_agents = 25
num_obstables = 2
num_targets = 100 

env = GameOfDronesEnv(num_agents, num_obstables, num_targets)
env.reset()
test_action = np.array([-np.ones(num_agents) * .85, 
                        np.zeros(num_agents), 
                        -np.ones(num_agents) * .15]).T * 200

obs_t, obs_t_Plus1, reward_t, done_t = env.step(test_action) # Expecting an np array that is (num_agents, 3)

# -------- Neural network parameters -------- #
hidden_size = 64
lr = 0.001

# -------- Environment -------- #
replay_buffer_max_size = 500

#%% Initialize the enviroment 
obs_size = obs_t.size

act_size = test_action.size                   # x,y,z directions of the propulsion force for each agent  

#%% Initialize the actor, critic, and target networks 
actor = Actor(obs_size, hidden_size, act_size)
actor_target = Actor(obs_size, hidden_size, act_size)
critic = Critic(obs_size + act_size, hidden_size, act_size)
critic_target = Critic(obs_size + act_size, hidden_size, act_size)

# Initialize the loss functions and the optimizers 
# Define the loss function and optimizer for the critic 
critic_loss_function = torch.nn.MSELoss()
critic_optimizer     = torch.optim.Adam(critic.parameters())

# Define the optimizer for the actor 
actor_optimizer      = torch.optim.Adam(actor.parameters())

# We initialize the target networks as copies of the original networks
for target_param, param in zip(actor_target.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

#%% Initiliaze the Replay Buffer 
ReplayBuffer = Memory(replay_buffer_max_size)

#%% Main training loop 

# Allocate memory for saving variables 
avg_critic_loss     = np.zeros((num_episodes))
avg_actor_loss      = np.zeros((num_episodes))
avg_episode_reward  = np.zeros((num_episodes))

for episode in range(num_episodes):

    # Resent the environment for each episode
    print("Episode #%d" % episode)
    env.reset(seed=1) # Or alternatively env.reset(seed=episode)
    # env.visualize()

    # Allocate memory for saving variables throughout each episode
    critic_losses = np.zeros((num_time_steps_per_episode))
    actor_losses  = np.zeros((num_time_steps_per_episode))
    rewards       = np.zeros((num_time_steps_per_episode))

    for t in range(num_time_steps_per_episode):

        # Find out where you currently are 
        obs_t = env.get_current_observation()

        # env.visualize()
        
        # Use the actor to predict an action from the current state
        a_t = actor.forward(utils.from_numpy(obs_t)) # the actor's forward pass needs a torch.Tensor

        # Convert action to numpy array 
        a_t = utils.to_numpy(a_t)
        # a_t = test_action.flatten()
        # a_t = 2*np.random.random(num_agents*3)-1
        obs_t, obs_t_Plus1, reward_t, done_t = env.step(a_t) # the env needs a numpy array

        if episode == num_episodes - 1:
            env.visualize()
            
        ReplayBuffer.push(obs_t, obs_t_Plus1, a_t, reward_t, done_t) # All pushed into the ReplayBuffer need to be numpy arrays

        if len(ReplayBuffer) > batch_size: # As soon as the ReplayBuffer has accumulated enough memory, perform RL

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
            # Get current Q-estimates
            Q_t_B = critic.forward(obs_t_B,a_t_B)
            # Use actor to predict next action given next states
            a_t_plus1_B = actor_target.forward(obs_t_Plus1_B)
            # Define the target Q's given the reward and the discounted next Q's
            target_Qs = reward_t_B + gamma * critic_target.forward(obs_t_Plus1_B, a_t_plus1_B)
            # NOTE: Regarding the line above. There is a predicted Q value for every action. But there is only one reward for each group of actions. 
            # Assert that shapes of the estimated Q's and the target Q's are the same 
            assert Q_t_B.size() == target_Qs.size()
            # Feed the estimate Q values and target Q values
            critic_loss          = critic_loss_function(Q_t_B, target_Qs)
            # Zero out the gradients and take a step of gradient descent
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ------ 2.) Update the actor ------ #
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

    avg_critic_loss[episode]    = np.mean(critic_losses)
    avg_actor_loss[episode]     = np.mean(actor_losses)
    avg_episode_reward[episode] = np.mean(rewards)


#%% Plot results 
import matplotlib.pyplot as plt

# Episodes 
episodes = np.arange(num_episodes)

# Create a figure with three subpanels
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.plot(episodes, avg_critic_loss,'g')
plt.yscale('log')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Critic Loss', fontsize=20)

plt.subplot(1,3,2)
plt.plot(episodes, avg_actor_loss, 'r')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Actor Loss', fontsize=20)

plt.subplot(1,3,3)
plt.plot(episodes, avg_episode_reward)
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.savefig('output/results_episode{}_agents{}_targets{}.png'.format(num_episodes, num_agents, num_targets))
# plt.show()