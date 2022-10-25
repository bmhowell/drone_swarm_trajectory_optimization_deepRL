"""
deep rl environment for game of drones

author: bhowell@berkeley.edu
date: 10/22
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
from matplotlib import rc

class GameOfDronesEnv():
    
    def __init__(self, n_agents=15, n_obstacles=25, n_targets=100):
        # CONSTANTS - dynamics and integration parameters
        self.Ai = 1                                         # |   m^2   | SCALAR - Agent characteristic area
        self.Cdi = 0.25                                     # |  none   | SCALAR - Agent coefficient of drag
        self.mi = 10                                        # |   kg    | SCALAR - Agent mass
        self.Fpi = 200                                      # |    N    | SCALAR - Propulsion force direction
        self.vA = np.array([-0.2, 0.2, 0.5])                # |   m/s   | VECTOR - Velocity of surrounding air
        self.rhoA = 1.225                                   # | kg/m^3  | SCALAR - Air density
        self.dt = 0.2                                       # |    s    | SCALAR - Time step
        self.tf = 60                                        # |    s    | SCALAR - Max task time

        self.total_steps = int(np.ceil(self.tf / self.dt))  # |  none   | SCALAR - total iterations in sim.

        # CONSTANTS - Objects and Interaction Parameters
        self.agent_sight = 5                                # |    m    | SCALAR - Target mapping distance 
        self.crash_range = 2                                # |    m    | SCALAR - Agent collision distance
        self.obs_distance = 25                              # |    m    | SCALAR - Observable distance

        self.nA0 = n_agents                                 # |   none  | SCALAR - Number of initial agents
        self.nO0 = n_obstacles                              # |   none  | SCALAR - Number of initial obstacles
        self.nT0 = n_targets                                # |   none  | SCALAR - Number of initial targets
        self.nA  = n_agents                                 # |   none  | SCALAR - Number of agents
        self.nO  = n_obstacles                              # |   none  | SCALAR - Number of obstacles
        self.nT  = n_targets                                # |   none  | SCALAR - Number of targets

        # log initial values
        self.nA0 = n_agents
        self.nO0 = n_obstacles
        self.nT0 = n_targets

        # observation dimensions
        self.ad_obs = 8                                     # |   none  | SCALAR - observation dimension of agents
        self.td_obs = 5                                     # |   none  | SCALAR - observation dimension of obstacles
        self.od_obs = 3                                     # |   none  | SCALAR - observation dimension of targets

        # CONSTANTS - Boundaries
        self.xB = 100                                       # |    m    | SCALAR - x boundary: target
        self.yB = 100                                       # |    m    | SCALAR - y Boundary: target
        self.zB = 10                                        # |    m    | SCALAR - z boundary: target
        self.xMax = 150                                     # |    m    | SCALAR - x boundary: lost drone
        self.yMax = 150                                     # |    m    | SCALAR - y boundary: lost drone
        self.zMax = 60                                      # |    m    | SCALAR - x boundary: lost drone

        # REWARD 
        self.w1 = 70                                        # Weight of mapping in net cost
        self.w2 = 10                                        # Weight of time usage in net cost
        self.w3 = 20                                        # Weight of agent losses in net cost

        # AGENTS, TARGETS, OBSTACLES
        self._agent_state = None

        self._target_velocity = None
        self._obstacle_velocity = None

    def get_current_state(self):
        return self._agent_state

    def get_current_target(self):
        return self._target_position
    
    def _get_active_objects(self, obj: np.ndarray) -> np.ndarray:
        if len(obj[0, :]) == self.ad_obs:
            return np.where(obj[:, 6] == True)[0]
        else:
            return np.where(obj[:, 3] == True)[0]

    def reset(self, seed=None):

        # housekeeping
        if seed != None:
            np.random.seed(seed)
        self.counter = 0
        self.done = False

        # initialize the drones (agents) along the sideline of domain
        self._agent_state = np.empty((self.nA0, self.ad_obs))
        xM = (self.xMax - 0.05 * self.xMax) * np.ones(self.nA)
        yM = np.linspace(-self.yMax + 0.05 * self.yMax, self.yMax - 0.05 * self.yMax, self.nA)
        zM = np.zeros(self.nA)
        self._agent_state[:, :3] = np.array([xM, yM, zM], dtype=np.float64).T
        self._agent_state[:, 3:6] = np.zeros((self.nA0, 3))
        self._agent_state[:, 6] = True
        self._agent_state[:, 7] = False

        # randomly position targets throughout the domain
        self._target_position = np.empty((self.nT0, self.td_obs))
        xT = np.random.rand(self.nT) * (2 * self.xB) - self.xB
        yT = np.random.rand(self.nT) * (2 * self.yB) - self.yB
        zT = np.random.rand(self.nT) * (2 * self.zB) - self.zB
        self._target_position[:, :3] = np.array([xT, yT, zT], dtype=np.float64).T
        self._target_position[:, 3] = True                           # one hot encoding for active
        self._target_position[:, 4] = False                          # one hot encoding for captured

        self.active_agents = self._get_active_objects(self._agent_state)
        self.active_targets = self._get_active_objects(self._target_position)

        # randomly position obstacles throughout the domain
        xO = np.random.rand(self.nO) * (2 * self.xB) - self.xB
        yO = np.random.rand(self.nO) * (2 * self.yB) - self.yB
        zO = np.random.rand(self.nO) * (2 * self.zB) - self.zB
        self._obstacle_position = np.array([xO, yO, zO], dtype=np.float64).T

        # PAIRWISE DISTANCE ARRAYS
        # declace arrays for differences between targets, agents and obstacles (vector)
        self.atDiff = np.empty((self.nA0, self.nT0, 3))              # agent to target distance (nA, nT, 3)
        self.aaDiff = np.empty((self.nA0, self.nA0, 3))              # agent to agent distance (nA, nA, 3)
        self.aoDiff = np.empty((self.nA0, self.nO0, 3))              # agent to obstacle distance (nA, nO, 3)

        # declace arrays for distances between targets, agents and obstacles (scalar)
        self.atDist = np.empty((self.nA0, self.nT0))                 # agent to target distance   (nA, nT)
        self.aaDist = np.empty((self.nA0, self.nA0))                 # agent to agent distance    (nA, nA)
        self.aoDist = np.empty((self.nA0, self.nO0))                 # agent to obstacle distance (nA, nO)
        
    def _get_pairwise_distances(self, active_agents: np.ndarray, active_targets: np.ndarray):
        """
            TODO
                -finish vectorising
        """
        for agent in active_agents:
            self.atDiff[agent, active_targets, :] = (self._target_position[active_targets, :3] - self._agent_state[agent, :3])
            self.aaDiff[agent, active_agents, :] = (self._agent_state[active_agents, :3] - self._agent_state[agent, :3])
            self.aoDiff[agent, :, :] = (self._obstacle_position - self._agent_state[agent, :3])
            self.aaDiff[agent, agent, :] = np.nan

            self.atDist[agent, active_targets] = np.linalg.norm(self._agent_state[agent, :3] - self._target_position[active_targets, :3], ord=2, axis=1)
            self.aaDist[agent, active_agents] = np.linalg.norm(self._agent_state[agent, :3] - self._agent_state[active_agents, :3], ord=2, axis=1)
            self.aoDist[agent, :] = np.linalg.norm(self._agent_state[agent, :3] - self._obstacle_position, ord=2, axis=1)
            self.aaDist[agent, agent] = np.nan
        
    def step(self, action):
        #####################################################################
        #------------------------ compute dynamics --------------------------
        # - get action force
        # - compute drag force
        # - step with forward euler
        #####################################################################

        # map action to propulsion force vector --- WHAT TO DO HERE
        f_prop = np.zeros((self.nA0, 3))
        f_prop[self.active_agents, :] = action[self.active_agents, :]

        # compute drag force applies to all agents (wind)
        v_norm = np.linalg.norm(self.vA - self._agent_state[:, 3:6], 2, axis=1)[:, np.newaxis]
        f_drag = 1. / 2. * self.rhoA * self.Cdi * self.Ai * v_norm * (self.vA - self._agent_state[self.active_agents, 3:6])

        f_total = f_drag + f_prop

        # step with forward euler
        self._agent_state[self.active_agents, 3:6] = self._agent_state[self.active_agents, 3:6] + self.dt * f_total / self.mi
        self._agent_state[self.active_agents, :3] = self._agent_state[self.active_agents, :3] + self.dt * self._agent_state[self.active_agents, 3:6]
        self.counter += 1
        
        #####################################################################
        # check for crashes, lost agents, targets achieved, etc.
        # nA = 15  # |   none  | SCALAR - Number of initial agents
        # nO = 25  # |   none  | SCALAR - Number of obstacles
        # nT = 100  # |   none  | SCALAR - Number of initial targets
        #####################################################################

        # get pairwise difference and distance matrices for all active agents
        self._get_pairwise_distances(self.active_agents, self.active_targets)

        # check if agents are in the range of the targets, obstacles, agents, our boundaries
        atHit = np.where(self.atDist[self.active_agents] < self.agent_sight)[1]
        aoHit = np.where(self.aoDist[self.active_agents] < self.crash_range)[0]
        aaHit = np.where(self.aaDist[self.active_agents] < self.crash_range)[0]

        # check for lost agents
        xLost = np.where(np.abs(self._agent_state[self.active_agents, 0]) > self.xMax)[0]
        yLost = np.where(np.abs(self._agent_state[self.active_agents, 1]) > self.yMax)[0]
        zLost = np.where(np.abs(self._agent_state[self.active_agents, 2]) > self.zMax)[0]

        # return i indices of lost drones
        aLost = np.unique(np.hstack([xLost, yLost, zLost]))

        # index all targets mapped
        target_mapped = np.unique(atHit)

        # index all agents crashed (from agent, obstacle and lost
        mCrash = np.unique(np.hstack([aaHit, aoHit, aLost]))

        # update active and inactive agents
        self._agent_state[mCrash, 6] = False
        self._agent_state[mCrash, 7] = True
        self._target_position[target_mapped, 3] = False
        self._target_position[target_mapped, 4] = True

        self.nT = self.nT0 - len(target_mapped)
        self.nM = self.nA0 - len(mCrash)

        # get active agents and targets
        self.active_agents = self._get_active_objects(self._agent_state)
        self.active_targets = self._get_active_objects(self._target_position)

        # if all agents are lost, crashed, or eliminated, stop the simulation
        if self.nT <= 0 or self.nM <= 0 or self.counter == self.total_steps:
            self.done = True

        # compute reward - ** CONSIDER ADDING TIME
        n_mapped_targets = np.sum(self._target_position[:, 4])
        n_crashed_drones = np.sum(self._agent_state[:, 7])
        reward = n_mapped_targets / self.nT0 - n_crashed_drones / self.nA0 

        observation_array = np.hstack(
                                (self._agent_state[:, :6].flatten(),            # agent positions and velocities
                                 self._target_position[:, :3].flatten(),        # target positions
                                 self._obstacle_position.flatten(),             # obstacle position
                                 self._agent_state[:, 6:8].flatten(),           # one hot enoded crash or not crash
                                 self._target_position[:, 3:5].flatten()        # one hot encoded active or not active
                                 )
                            )

        return observation_array, reward, self.done
    
    def visualize(self):
        print('-- plotting configuration --')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self._agent_state[self.active_agents, 0], 
                   self._agent_state[self.active_agents, 1], 
                   self._agent_state[self.active_agents, 2], 
                   color='r')

        ax.scatter(self._target_position[self.active_targets, 0], 
                   self._target_position[self.active_targets, 1], 
                   self._target_position[self.active_targets, 2], 
                   color='g')

        ax.scatter(self._obstacle_position[:, 0], self._obstacle_position[:, 1], self._obstacle_position[:, 2], color='k')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=70., azim=40)
        ax.legend(['Obstacles', 'Targets', 'Agents'])
        plt.show()


if __name__ == "__main__":
    """
        TESTING
    """
    env = GameOfDronesEnv()
    env.reset()
    env.visualize()
    print('before: \n', env.get_current_state()[:, :3])
    test_action = np.array([-np.ones(15) * .85, 
                            np.zeros(15), 
                            -np.ones(15) * .15]).T * 200
    env.step(test_action)
    # print('after: \n', env.get_current_state()[:, :3])

    done = False
    counter = 0
    while not done:
        obs, reward, done = env.step(test_action)
        print('counter = ', counter)
        print('obs.shape: ', obs.shape)
        print('reward: ', reward)
        print('done: ', done)
        print('')
        counter += 1
    
    # print('\nafter: \n', env.get_current_state()[:, :3])
    env.visualize()
    print('DONE')


    



    




        

