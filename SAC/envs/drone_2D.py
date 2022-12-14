"""
deep rl environment for game of drones

author: bhowell@berkeley.edu
date: 10/22
updated by: rdhuff@berkeley.edu
date: 11/07
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class GameOfDronesEnv():
    
    def __init__(self, n_agents=15, n_obstacles=25, n_targets=100):
        # CONSTANTS - dynamics and integration parameters
        self.Ai = 1                                         # |   m^2   | SCALAR - Agent characteristic area
        self.Cdi = 0.25                                     # |  none   | SCALAR - Agent coefficient of drag
        self.mi = 10                                        # |   kg    | SCALAR - Agent mass
        self.Fpi = 200                                      # |    N    | SCALAR - Propulsion force direction
        self.vA = np.array([-0.2, 0.2])                # |   m/s   | VECTOR - Velocity of surrounding air
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

        # observation dimensions
        self.ad_obs = 6                                     # |   none  | SCALAR - observation dimension of agents
        self.td_obs = 4                                     # |   none  | SCALAR - observation dimension of obstacles
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
        
        self.reward = []                                    # adding to track reward over time for plotting

        # AGENTS, TARGETS, OBSTACLES
        self._agent_state = None
        self._target_velocity = None
        self._obstacle_velocity = None

    def get_current_observation(self):
        current_observation = np.hstack(
                                (self._agent_state[:, :2].flatten(),            # agent positions and velocities
                                 self._target_position[:, :2].flatten(),        # target positions
                                #  self._obstacle_position.flatten(),             # obstacle position
                                #  self._agent_state[:, 6:8].flatten(),           # one hot enoded crash or not crash
                                #  self._target_position[:, 3:5].flatten()        # one hot encoded active or not active
                                 )
        ) / self.xMax
        return current_observation 

    def get_current_state(self):
        return self._agent_state

    def get_current_target(self):
        return self._target_position
    
    def _get_active_objects(self, obj: np.ndarray) -> np.ndarray:
        if len(obj[0, :]) == self.ad_obs: 
            return np.where(obj[:, 4] == True)[0] 
        elif len(obj[0,:]) == self.td_obs:
            return np.where(obj[:, 2] == True)[0]
        else:
            assert len(obj[0,:]) == 1000

    def reset(self, seed=None, randomAgentInitialization=True, randomTargetInitialization=True):

        # housekeeping
        if seed != None:
            np.random.seed(seed)
        self.counter = 0
        self.done = False

        # initialize the drones (agents) along the sideline of domain or randomly in the space
        self._agent_state = np.empty((self.nA0, self.ad_obs))
        if randomAgentInitialization is not True:
            xM = (self.xMax - 0.05 * self.xMax) * np.ones(self.nA)
            yM = np.linspace(-self.yMax + 0.05 * self.yMax, self.yMax - 0.05 * self.yMax, self.nA)
        elif randomAgentInitialization is True:
            xM = np.random.rand(self.nA0) * (2 * self.xB) - self.xB
            yM = np.random.rand(self.nA0) * (2 * self.yB) - self.yB
        self._agent_state[:, :2] = np.array([xM, yM], dtype=np.float64).T
        self._agent_state[:, 2:4] = np.zeros((self.nA0, 2))
        self._agent_state[:, 4] = True
        self._agent_state[:, 5] = False

        # randomly position targets throughout the domain
        self._target_position = np.empty((self.nT0, self.td_obs))
        if randomTargetInitialization is not True:
            xT = np.linspace(-0.8*(self.xMax), 0.8*(self.xMax), self.nT)
            yT = np.linspace(0,0,self.nT)
        elif randomTargetInitialization is True: 
            xT = np.random.rand(self.nT0) * (2 * self.xB) - self.xB
            yT = np.random.rand(self.nT0) * (2 * self.yB) - self.yB
        self._target_position[:, :2] = np.array([xT, yT], dtype=np.float64).T
        self._target_position[:, 2] = True                           # one hot encoding for active
        self._target_position[:, 3] = False                          # one hot encoding for captured

        # Use the _get_active_objects function to return the indices of the active agents and targets
        self.active_agents = self._get_active_objects(self._agent_state)
        self.active_targets = self._get_active_objects(self._target_position)

        # randomly position obstacles throughout the domain
        xO = np.random.rand(self.nO) * (2 * self.xB) - self.xB
        yO = np.random.rand(self.nO) * (2 * self.yB) - self.yB
        self._obstacle_position = np.array([xO, yO], dtype=np.float64).T

        # PAIRWISE DISTANCE ARRAYS
        # declace arrays for differences between targets, agents and obstacles (vector)
        self.atDiff = np.empty((self.nA0, self.nT0, 2))              # agent to target distance (nA, nT, 3)
        self.aaDiff = np.empty((self.nA0, self.nA0, 2))              # agent to agent distance (nA, nA, 3)
        self.aoDiff = np.empty((self.nA0, self.nO0, 2))              # agent to obstacle distance (nA, nO, 3)

        # declace arrays for distances between targets, agents and obstacles (scalar)
        self.atDist = np.empty((self.nA0, self.nT0))                 # agent to target distance   (nA, nT)
        self.aaDist = np.empty((self.nA0, self.nA0))                 # agent to agent distance    (nA, nA)
        self.aoDist = np.empty((self.nA0, self.nO0))                 # agent to obstacle distance (nA, nO)
        
        self.reward = []
        
    def _get_pairwise_distances(self, active_agents: np.ndarray, active_targets: np.ndarray):
        """
            TODO
                -finish vectorising
        """
        for agent in active_agents:
            self.atDiff[agent, active_targets, :] = (self._target_position[active_targets, :2] - self._agent_state[agent, :2])
            self.aaDiff[agent, active_agents, :] = (self._agent_state[active_agents, :2] - self._agent_state[agent, :2])
            self.aoDiff[agent, :, :] = (self._obstacle_position - self._agent_state[agent, :2])
            self.aaDiff[agent, agent, :] = np.nan

            self.atDist[agent, active_targets] = np.linalg.norm(self._agent_state[agent, :2] - self._target_position[active_targets, :2], ord=2, axis=1)
            self.aaDist[agent, active_agents] = np.linalg.norm(self._agent_state[agent, :2] - self._agent_state[active_agents, :2], ord=2, axis=1)
            self.aoDist[agent, :] = np.linalg.norm(self._agent_state[agent, :2] - self._obstacle_position, ord=2, axis=1)
            self.aaDist[agent, agent] = np.nan
        
    def step(self, action):
        # ensure correct shape
        if action.shape != (self.nA0, 2):
            action = action.reshape(self.nA0, 2)
        
        assert action.shape == (self.nA0, 2), "Error: action reshaping not working"
        #####################################################################
        #------------------------ compute dynamics --------------------------
        # - get action force
        # - compute drag force
        # - step with forward euler
        #####################################################################
        
        # copy current observation
        current_observation = self.get_current_observation()

        # map action to propulsion force vector --- WHAT TO DO HERE

        # normalize action
        f_prop = np.zeros((self.nA0, 2))
        action /= np.linalg.norm(action, ord=2, axis=1)[:, None]
        action4step = action.copy()
        action *= self.Fpi
        
        f_prop[self.active_agents, :] = action[self.active_agents, :]

        # compute drag force applies to all agents (wind)
        v_norm = np.linalg.norm(self.vA - self._agent_state[:, 2:4], 2, axis=1)[:, np.newaxis]
        f_drag = 1. / 2. * self.rhoA * self.Cdi * self.Ai * v_norm * (self.vA - self._agent_state[:, 2:4])
 
        f_total = f_drag[self.active_agents, :] + f_prop[self.active_agents, :]

        # step with forward euler
        self._agent_state[self.active_agents, 2:4] = self._agent_state[self.active_agents, 2:4] + self.dt * f_total / self.mi
        self._agent_state[self.active_agents, :2] = self._agent_state[self.active_agents, :2] + self.dt * self._agent_state[self.active_agents, 2:4]
        # self._agent_state[self.active_agents, :2] = self._agent_state[self.active_agents, :2] + action4step * 3
        self.counter += 1
        
        #####################################################################
        # check for crashes, lost agents, targets achieved, etc.
        # nA = 15  #  |   none  | SCALAR - Number of initial agents
        # nO = 25  #  |   none  | SCALAR - Number of obstacles
        # nT = 100  # |   none  | SCALAR - Number of initial targets
        #####################################################################

        # get pairwise difference and distance matrices for all active agents
        self._get_pairwise_distances(self.active_agents, self.active_targets)

        # check if agents are in the range of the targets, obstacles, agents, our boundaries
#        atHit = np.where(self.atDist[self.active_agents] < self.agent_sight)[1]
#        aoHit = np.where(self.aoDist[self.active_agents] < self.crash_range)[0]
#        aaHit = np.where(self.aaDist[self.active_agents] < self.crash_range)[0] #these have same problem as xlost and ylost did
        
        atHit = np.where(self.atDist[:] < self.agent_sight)[1]
        aoHit = np.where(self.aoDist[:] < self.crash_range)[0]
        aaHit = np.where(self.aaDist[:] < self.crash_range)[0]

        # check for lost agents
        xLost = np.where(np.abs(self._agent_state[:, 0]) > self.xMax)[0]
        yLost = np.where(np.abs(self._agent_state[:, 1]) > self.yMax)[0]

        # return i indices of lost drones
        aLost = np.unique(np.hstack([xLost, yLost]))

        # index all targets mapped
        target_mapped = np.unique(atHit)

        # index all agents crashed (from agent, obstacle and lost
        mCrash = np.unique(np.hstack([aaHit, aoHit, aLost]))

        # update active and inactive agents
        self._agent_state[mCrash, 4] = False
        self._agent_state[mCrash, 5] = True
        self._target_position[target_mapped, 2] = False
        self._target_position[target_mapped, 3] = True

        # get active agents and targets
        self.active_agents = self._get_active_objects(self._agent_state)
        self.active_targets = self._get_active_objects(self._target_position)

        self.nT = len(self.active_targets) # self.nT0 - len(target_mapped)
        self.nA = len(self.active_agents)  # self.nA0 - len(mCrash)

        # if all agents are lost, crashed, or eliminated, stop the simulation
        if self.nT <= 0: 
            print('Target found!')
        if self.nT <= 0 or self.nA <= 0 or self.counter == self.total_steps:
            self.done = True

        # compute reward - ** CONSIDER ADDING TIME
        n_mapped_targets = np.sum(self._target_position[:, 3])
        n_crashed_drones = np.sum(self._agent_state[:, 5])
        # print('n_crashed_drones: ', n_crashed_drones)
        # reward = n_mapped_targets / self.nT0 - n_crashed_drones / self.nA0 
        # reward = reward_nT - reward_nA
        largest_possible_dist_at = np.sqrt((2*self.xMax)**2 + (2*self.yMax)**2)
        # print(self.atDist)
        # print(self.atDist.shape)
        if len(self.active_targets) > 0:
            reward = (- np.sum(np.amin(self.atDist[self.active_agents, :][:, self.active_targets], axis=1))
                  + largest_possible_dist_at * (n_mapped_targets / self.nT0) 
                  - largest_possible_dist_at * (n_crashed_drones / self.nA0) ) 
        else:
            reward = 1000
        
        self.reward.append(reward)

        # print(reward)

        next_observation = np.hstack(
                                (self._agent_state[:, :2].flatten(),            # agent positions and velocities
                                 self._target_position[:, :2].flatten(),        # target positions
                                #  self._obstacle_position.flatten(),             # obstacle position
                                #  self._agent_state[:, 6:8].flatten(),           # one hot enoded crash or not crash
                                #  self._target_position[:, 3:5].flatten()        # one hot encoded active or not active
                                 )
                            ) / self.xMax

        return current_observation, next_observation, reward, self.done
    
    def visualize(self, savePath='output'):
        fig = plt.figure(figsize=(8,15))
        ax = fig.add_subplot(311)
        ax.scatter(self._agent_state[self.active_agents, 0], 
                   self._agent_state[self.active_agents, 1], 
                   color='r', label='agents')

        ax.scatter(self._target_position[self.active_targets, 0], 
                   self._target_position[self.active_targets, 1], 
                   color='g', label='targets')

        ax.scatter(self._obstacle_position[:, 0], self._obstacle_position[:, 1], color='k', label='obstacles')
        
        #adding for accuracy checks:
        n = np.arange(self.nA)
        for i, txt in enumerate(n):
            ax.annotate(txt, (self._agent_state[self.active_agents, 0][i], self._agent_state[self.active_agents, 1][i]))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axis('equal')
        ax.set_ylim([-200, 200])
        ax.set_xlim([-200, 200])
        
        #adding subplot to display atDist
        ax1 = fig.add_subplot(312)
        ax1.matshow(self.atDist)
        for (i, j), z in np.ndenumerate(self.atDist):
            ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
            
        ax2 = fig.add_subplot(313)
        ax2.plot(self.reward)
        
        plt_savePath = os.path.join( savePath, 'drone_{}.png'.format(self.counter) )
        plt.savefig(plt_savePath)
        plt.close()


if __name__ == "__main__":
    """
        TESTING
    """
    env = GameOfDronesEnv(2,0,2)
    env.reset()
    env.visualize()
    print('before: \n', env.get_current_state()[:, :2])
    test_action = np.array([-np.ones(2) * .85, 
                            np.zeros(2)]).T * 200
    env.step(test_action)
    # print('after: \n', env.get_current_state()[:, :3])

    done = False
    counter = 0
    while not done:
        curr_obs, next_obs, reward, done = env.step(test_action)
        print('counter = ', counter)
        print('curr_obs.shape: ', curr_obs.shape)
        print('next_obs.shape: ', next_obs)
        print('reward: ', reward)
        print('done: ', done)
        print('')
        env.visualize()
        counter += 1
    
    # print('\nafter: \n', env.get_current_state()[:, :3])
    env.visualize()
    print('DONE')

