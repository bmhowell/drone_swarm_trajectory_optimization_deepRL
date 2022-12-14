import argparse
import datetime
#import gym
import numpy as np
import itertools
import torch
from SAC.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from SAC.replay_memory import ReplayMemory
import gym
from gym import spaces

from envs.drone_2D import *
from infrastructure.utils import pt_normalize_actions_2D
from infrastructure.utils import from_numpy
from infrastructure.utils import to_numpy


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--num_tests', type=int, default=10, metavar='N',
                    help='number of tests (default: 10)')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--video', type=bool, default=False,
                    help='video')
args = parser.parse_args()

# -------- Environment -------- #
num_agents = 2
num_obstacles = 0
num_targets = 1

#obs_size = int(num_agents*2 + num_targets*2 + num_obstacles*2) # int(num_agents*2*3 + num_agents*2 + num_obstables * 3 + num_targets * 5)
#obs_size = 2*int(num_agents*2 + num_targets*2 + num_obstacles*2) # if you include one hot encoding in obs
obs_size = 2*int(num_agents*3 + num_targets*2 + num_obstacles*2) # if you include one hot encoding and velocities in obs
act_size = num_agents*2 # x,y,z directions of the propulsion force for each agent
#%% Initialize the enviroment
env = GameOfDronesEnv(num_agents, num_obstacles*2, num_targets)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
action_space = spaces.Box(-1*np.ones(act_size),np.ones(act_size),dtype=np.float32)
#agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent = SAC(obs_size, action_space, args)
ckpt_path = "runs/2022-12-13_00-58-02_SAC_drone_2D_Gaussian_autotune/checkpoints/sac_checkpoint_drone_2D_10500"
agent.load_checkpoint(ckpt_path, evaluate=True)


#Tesnorboard
#path = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
path = "runs/2022-12-13_00-58-02_SAC_drone_2D_Gaussian_autotune"
writer = SummaryWriter(path)
#print(path)
#
## Memory
#memory = ReplayMemory(args.replay_size, args.seed)
#
## Training Loop
#total_numsteps = 0
#updates = 0
#
#for i_episode in itertools.count(1):
#    episode_reward = 0
#    episode_steps = 0
#    done = False
##    state = env.reset()
#    env.reset()
#    state = env.get_current_observation()
#
#    while not done:
#        if args.start_steps > total_numsteps:
##            action = env.action_space.sample()  # Sample random action
#            action = pt_normalize_actions_2D(from_numpy(np.random.rand(act_size)))
#        else:
#            action = agent.select_action(state)  # Sample action from policy
#
##        print('action',action)
#        if len(memory) > args.batch_size:
#            # Number of updates per step in environment
#            for i in range(args.updates_per_step):
#                # Update parameters of all the networks
#                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
#
#                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
#                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
#                writer.add_scalar('loss/policy', policy_loss, updates)
#                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
#                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
#                updates += 1
#
#        try: #MARK this is janky
#            action = to_numpy(action)
#        except:
#            pass
#
##        next_state, reward, done, _ = env.step(action) # Step
#        state, next_state, reward, done = env.step(action)
#        episode_steps += 1
#        total_numsteps += 1
#        episode_reward += reward
#
#        # Ignore the "done" signal if it comes from hitting the time horizon.
#        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
##        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
#        mask = float(not done)
#        #we don't have a max num of time steps defined within env right? #MARK
#
#        memory.push(state, action, reward, next_state, mask) # Append transition to memory
#
#        state = next_state
#
#    if total_numsteps > args.num_steps:
#        break
#
##    print('episode_reward', episode_reward)
#    writer.add_scalar('reward/train', episode_reward, i_episode)
#    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

#    if i_episode % 10 == 0 and args.eval is True:
#        avg_reward = 0.
#        episodes = 10
#        for _  in range(episodes):
##            state = env.reset()
#            env.reset()
#            state = env.get_current_observation()
#            episode_reward = 0
#            done = False
#            while not done:
##                print('state',state)
#                action = agent.select_action(state, evaluate=True)
#
#                state, next_state, reward, done = env.step(action)
##                next_state, reward, done, _ = env.step(action)
##                print(reward)
#                episode_reward += reward
#
#                state = next_state
#            avg_reward += episode_reward
#        avg_reward /= episodes
#
#
#        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
##        writer.close()
#
#        print("----------------------------------------")
#        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
#        print("----------------------------------------")

for i_episode in range(args.num_tests):
    env.reset()
    state = env.get_current_observation()
    done = False
    t = 0
    os.makedirs(path+'/test_episode'+str(i_episode))
    while not done:
        action = agent.select_action(state, evaluate=True)

        state, next_state, reward, done = env.step(action)
#                next_state, reward, done, _ = env.step(action)
#                print(reward)

#            env.visualize(path+'/episode'+str(i_episode)+'/'+str(t)+'.png')
        env.visualize(path+'/test_episode'+str(i_episode)+'/')
        state = next_state
        t += 1
        
#        avg_reward += episode_reward
#    avg_reward /= episodes



#env.close()




#
##%% Inputs
## ----------- NOTE: Change this section to use such that it may be ----------- #
## -----------       run from the command line                      ----------- #
#
## -------- Testing -------- #
#num_episodes = 10
#num_time_steps_per_episode = 300
#
## -------- Environment -------- #
#num_agents = 1
#num_obstables = 0
#num_targets = 1
#
#obs_size = int(num_agents*2 + num_targets*2) # int(num_agents*2*3 + num_agents*2 + num_obstables * 3 + num_targets * 5)
#act_size = num_agents*2 # x,y,z directions of the propulsion force for each agent
#
## -------- Neural network parameters -------- #
#path2actor = "runs/2D_offlineRL_gameOfDrones_2022_11_09-16_35_08/actor.pth"
#hidden_size = 256
#
## -------- Logging -------- #
#visualizationOneInNrollouts = 1
#logdir = 'runs'
#exp_name = '2D_testing_offlineRL_before'
#now = datetime.now()
#savePath = exp_name + '_' + now.strftime("%Y_%m_%d-%H_%M_%S")
#tensorboardPath = os.path.join(logdir, savePath)
#writer = SummaryWriter(tensorboardPath)
#
##%% Initialize the enviroment
#env = GameOfDronesEnv(num_agents, num_obstables, num_targets)
#
##%% Initialize the actor and critic networks
#actor = Actor(obs_size, hidden_size, act_size)
#critic = Critic(obs_size + act_size, hidden_size, 1) # 1 output for the Q-value
#
##%% Load the trained policy
#actor.load_state_dict(torch.load(path2actor))
#
##%% Now that the actor and critic have been trained, let's apply them to new environments
#print('Training complete. Now testing the trained actor and critic on new environments')
#for episode in range(num_episodes):
#
#    # Reset the environment for each episode
#    print("Episode #%d" % episode)
#    env.reset(seed=episode, randomAgentInitialization=True, randomTargetInitialization=False) # Or alternatively env.reset(seed=episode)
#
#    for t in range(num_time_steps_per_episode):
#
#        # Find out where you currently are
#        obs_t = env.get_current_observation()
#        # Use the actor to predict an action from the current state
#        a_t = actor.forward(utils.from_numpy(obs_t)) # the actor's forward pass needs a torch.Tensor
#        # Convert action to numpy array
#        a_t = utils.to_numpy(a_t)
#        # Take a step in the environment
#        obs_t, obs_t_Plus1, reward_t, done_t = env.step(a_t) # the env needs a numpy array
#
#        # Visualize one in every n rollouts
#        if episode % visualizationOneInNrollouts == 0:
#            if t == 0:
#                print("Visualizing episode #%d" % episode)
#            visualizationPath = os.path.join(tensorboardPath,'episode{}'.format(episode))
#            if not os.path.exists(visualizationPath):
#                os.makedirs(visualizationPath)
#            env.visualize(savePath=visualizationPath)
#
#        if done_t is True:
#            writer.add_scalar('debug/rollout_length', t, episode)
#            print('Episode done')
#            break
