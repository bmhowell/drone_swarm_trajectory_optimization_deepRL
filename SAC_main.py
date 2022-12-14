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
import json

from envs.drone_2D import *
from infrastructure.utils import pt_normalize_actions_2D
from infrastructure.utils import from_numpy
from infrastructure.utils import to_numpy

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="drone_2D",
                    help='Mujoco Gym environment (default: drone_2D)')
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
parser.add_argument('--batch_size', type=int, default=10000, metavar='N',
                    help='batch size (default: 10000)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--num_eps', type=int, default=10000, metavar='N',
                    help='maximum number of episodes (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--steps_btwn_updates', type=int, default=100, metavar='N',
                    help='model updates per simulator step (default: 1)')
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
                    
#environment variables:
parser.add_argument('--num_agents', type=int, default=2, metavar='N',
                    help='number of agents/drones (default: 1)')
parser.add_argument('--num_obstacles', type=int, default=0, metavar='N',
                    help='number of obstacles (default: 0)')
parser.add_argument('--num_targets', type=int, default=1, metavar='N',
                    help='number of targets (default: 1)')
                    
#if starting from an existing checkpoint:
parser.add_argument('--load_checkpoint', type=bool, default=False,
                    help='whether it should load a checkpoint or not, default False')
parser.add_argument('--buffer_path', default="",
                    help='path to memory buffer checkpoint')
parser.add_argument('--checkpoint_path', default="",
                    help='path to model checkpoint')

args = parser.parse_args()

## Environment
## env = NormalizedActions(gym.make(args.env_name))
#env = gym.make(args.env_name)
#env.seed(args.seed)
#env.action_space.seed(args.seed)

# -------- Environment -------- #
num_agents = args.num_agents
num_obstacles = args.num_obstacles
num_targets = args.num_targets

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
if args.load_checkpoint == True:
    agent.load_checkpoint(args.checkpoint_path, evaluate=False)

#Tesnorboard
path = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
                                                             
os.makedirs(path+'/checkpoints')
    
writer = SummaryWriter(path)
print(path)

with open(path+'/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
if args.load_checkpoint == True:
    memory.load_buffer(args.buffer_path)

print('/n/n',len(memory),'/n/n')

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
#    state = env.reset()
    env.reset(seed = i_episode)
    state = env.get_current_observation()

    while not done:
        if args.start_steps > total_numsteps:
#            action = env.action_space.sample()  # Sample random action
            action = pt_normalize_actions_2D(from_numpy(np.random.rand(act_size)))
        else:
            action = agent.select_action(state)  # Sample action from policy

#        print('action',action)
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            if total_numsteps % args.steps_btwn_updates == 0:
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
                
        try: #MARK this is janky
            action = to_numpy(action)
        except:
            pass
            
#        next_state, reward, done, _ = env.step(action) # Step
        state, next_state, reward, done = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        
        writer.add_scalar('avg_reward/train', episode_reward/episode_steps, i_episode)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = float(not done)
        #we don't have a max num of time steps defined within env right? #MARK

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

#if want to break by number of steps run:
#    if total_numsteps > args.num_steps:
#        break

#if want to break based on number of episodes run:
    if i_episode > args.num_eps:
        break

#    print('episode_reward', episode_reward)
#    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
#            state = env.reset()
            env.reset()
            state = env.get_current_observation()
            episode_reward = 0
            done = False
            while not done:
#                print('state',state)
                action = agent.select_action(state, evaluate=True)

                state, next_state, reward, done = env.step(action)
#                next_state, reward, done, _ = env.step(action)
#                print(reward)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
#        writer.close()

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    if i_episode % 500 == 0 and args.video is True:
        env.reset()
        state = env.get_current_observation()
        done = False
        t = 0
        os.makedirs(path+'/episode'+str(i_episode))
        while not done:
            action = agent.select_action(state, evaluate=True)

            state, next_state, reward, done = env.step(action)
#                next_state, reward, done, _ = env.step(action)
#                print(reward)

#            env.visualize(path+'/episode'+str(i_episode)+'/'+str(t)+'.png')
            env.visualize(path+'/episode'+str(i_episode)+'/')
            state = next_state
            t += 1
            
        ckpt_path = path+'/checkpoints/sac_checkpoint_{}_{}'.format(args.env_name, i_episode)
        agent.save_checkpoint(args.env_name, ckpt_path = ckpt_path) #saving model states
        
        ckpt_path = path+'/checkpoints/sac_buffer_{}_{}'.format(args.env_name, i_episode)
        memory.save_buffer(args.env_name,save_path = ckpt_path) #saving buffer
        
#        avg_reward += episode_reward
#    avg_reward /= episodes



#env.close()

