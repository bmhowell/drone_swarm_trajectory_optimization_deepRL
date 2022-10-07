import gym
from gym import spaces
import pygame
import numpy as np
from collections import OrderedDict


class GameOfDronesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_agents=15, n_obstacles=25, n_targets=100):

        # CONSTANTS - dynamics and integration parameters
        self.Ai = 1                                         # |   m^2   | SCALAR - Agent characteristic area
        self.Cdi = 0.25                                     # |  none   | SCALAR - Agent coefficient of drag
        self.mi = 10                                        # |   kg    | SCALAR - Agent mass
        self.Fpi = 200                                      # |    N    | SCALAR - Propulsion force direction
        self.vA = np.array([0, 0, 0])                       # |   m/s   | VECTOR - Velocity of surrounding air
        self.rhoA = 1.225                                   # | kg/m^3  | SCALAR - Air density
        self.dt = 0.2                                       # |    s    | SCALAR - Time step
        self.tf = 60                                        # |    s    | SCALAR - Max task time
        self.counter = 0                                    # |  none   | SCALAR - Iteration counter

        self.total_steps = int(np.ceil(self.tf / self.dt))  # |  none   | SCALAR - total iterations in sim.

        # CONSTANTS - Objects and Interaction Parameters
        self.agent_sight = 5                                # |    m    | SCALAR - Target mapping distance 
        self.crash_range = 2                                # |    m    | SCALAR - Agent collision distance
        self.obs_distance = 25                              # |    m    | SCALAR - Observable distance 
        self.nA = n_agents                                  # |   none  | SCALAR - Number of initial agents
        self.nO = n_obstacles                               # |   none  | SCALAR - Number of obstacles
        self.nT = n_targets                                 # |   none  | SCALAR - Number of initial targets

        # log initial values
        self.nA0 = n_agents
        self.nO0 = n_obstacles
        self.nT0 = n_targets

        # CONSTANTS - Boundaries
        self.xB = 100                                       # |    m    | SCALAR - x boundary: target
        self.yB = 100                                       # |    m    | SCALAR - y Boundary: target
        self.zB = 10                                        # |    m    | SCALAR - z boundary: target
        self.xMax = 150                                     # |    m    | SCALAR - x boundary: lost drone
        self.yMax = 150                                     # |    m    | SCALAR - y boundary: lost drone
        self.zMax = 60                                      # |    m    | SCALAR - x boundary: lost drone

        # Box function making me discretise every goddamn agent, target and obstacle to a dictionary
        total_observations = {}
        for i in range(self.nA):
            total_observations['agent_{}'.format(i)] = spaces.Box(low=-np.array([np.float64(self.xMax), np.float64(self.yMax), np.float64(self.zMax)]).T, 
                                                                  high=np.array([np.float64(self.xMax), np.float64(self.yMax), np.float64(self.zMax)]).T,
                                                                  shape=(3,),
                                                                  dtype=np.float64)
        for i in range(self.nT):
            total_observations['target_{}'.format(i)] = spaces.Box(low=-np.array([np.float64(self.xMax), np.float64(self.yMax), np.float64(self.zMax)]).T, 
                                                                   high=np.array([np.float64(self.xMax), np.float64(self.yMax), np.float64(self.zMax)]).T,
                                                                   shape=(3,),
                                                                   dtype=np.float64)
        for i in range(self.nO):
            total_observations['obstacle_{}'.format(i)] = spaces.Box(low=-np.array([np.float64(self.xMax), np.float64(self.yMax), np.float64(self.zMax)]).T, 
                                                                   high=np.array([np.float64(self.xMax), np.float64(self.yMax), np.float64(self.zMax)]).T,
                                                                   shape=(3,),
                                                                   dtype=np.float64)



        self.observation_space = spaces.Dict(total_observations)

        # FIX THIS
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environAent is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        # gym DEMANDS every dictionary item to be squeezed
        total_observations = {}
        for i in range(self.nA):
            total_observations['agent_{}'.format(i)] = self._agent_position[i, :].squeeze()
        for i in range(self.nT):
            total_observations['target_{}'.format(i)] = self._target_position[i, :].squeeze()
        for i in range(self.nO):
            total_observations['obstacle_{}'.format(i)] = self._obstacle_position[i, :].squeeze()
        return total_observations

    def _get_info(self):
        return {
            "agent_to_target_distance": self.atDist,
            "agent_to_agent_distance": self.aaDist,
            "agent_to_obstacle_distance": self.aoDist
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        #################################################################################
        # INITIALIZE ENVIRONMENT
        # randomly position obstacles throughout the domain
        xO = np.random.rand(self.nO) * (2 * self.xB) - self.xB
        yO = np.random.rand(self.nO) * (2 * self.yB) - self.yB
        zO = np.random.rand(self.nO) * (2 * self.zB) - self.zB
        self._obstacle_position = np.array([xO, yO, zO], dtype=np.float64).T

        # randomly position targets throughout the domain
        xT = np.random.rand(self.nT) * (2 * self.xB) - self.xB
        yT = np.random.rand(self.nT) * (2 * self.yB) - self.yB
        zT = np.random.rand(self.nT) * (2 * self.zB) - self.zB
        self._target_position = np.array([xT, yT, zT], dtype=np.float64).T

        # initialize the drones (agents) along the sideline of domain
        xM = (self.xMax - 0.05 * self.xMax) * np.ones(self.nA)
        yM = np.linspace(-self.yMax + 0.05 * self.yMax, self.yMax - 0.05 * self.yMax, self.nA)
        zM = np.zeros(self.nA)
        self._agent_position = np.array([xM, yM, zM], dtype=np.float64).T

        self._agent_velocity = np.zeros((self.nA, 3))

        # PAIRWISE DISTANCE ARRAYS
        # compute differences between targets, agents and obstacles (vector)
        self.atDiff = np.zeros((len(self._agent_position[:, 0]), len(self._target_position[:, 0]), 3))              # agent to target distance (nA, nT, 3)
        self.aaDiff = np.zeros((len(self._agent_position[:, 0]), len(self._agent_position[:, 0]), 3))               # agent to agent distance (nA, nA, 3)
        self.aoDiff = np.zeros((len(self._agent_position[:, 0]), len(self._obstacle_position[:, 0]), 3))            # agent to obstacle distance (nA, nO, 3)

        # compute distances between targets, agents and obstacles (scalar)
        self.atDist = np.zeros((len(self._agent_position[:, 0]), len(self._target_position[:, 0])))                 # agent to target distance   (nA, nT)
        self.aaDist = np.zeros((len(self._agent_position[:, 0]), len(self._agent_position[:, 0])))                  # agent to agent distance    (nA, nA)
        self.aoDist = np.zeros((len(self._agent_position[:, 0]), len(self._obstacle_position[:, 0])))               # agent to obstacle distance (nA, nO)
        
        # check each agent (agent -> nA)
        for j in range(len(self._agent_position[:, 0])):
            self.atDiff[j, :, :] = (self._target_position - self._agent_position[j])                                 # (nA x nT x 3)
            self.aaDiff[j, :, :] = (self._agent_position - self._agent_position[j])                                 # (nA x nA x 3)
            self.aoDiff[j, :, :] = (self._obstacle_position - self._agent_position[j])                               # (nA x nO x 3)
            self.aaDiff[j, j, :] = np.nan

            self.atDist[j, :] = np.linalg.norm(self._agent_position[j] - self._target_position, ord=2, axis=1)       # (nT x 3) --norm--> (nT x 1)  ∈ (nA, nT)
            self.aaDist[j, :] = np.linalg.norm(self._agent_position[j] - self._agent_position, ord=2, axis=1)       # (nA x 3)  --norm--> (nA x 1) ∈ (nA, nA)
            self.aoDist[j, :] = np.linalg.norm(self._agent_position[j] - self._obstacle_position, ord=2, axis=1)     # (nO x 3)  --norm--> (nO x 1) ∈ (nA, nO)

            # don't count distances of agents with themselves
            self.aaDist[j, j] = np.nan
        
        #################################################################################

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        #####################################################################
        #------------------------ compute dynamics --------------------------
        # - get action force
        # - compute drag force
        # - step with forward euler
        #####################################################################
        
        # map action to propulsion force vector --- WHAT TO DO HERE
        fProp = action
        # fProp = self._action_to_propulsion[action]
        
        # compute drag force: 
        vNormDiff = np.linalg.norm(self.vA - self._agent_velocity, 2, axis=1)[:, np.newaxis]
        fDrag = 1. / 2. * self.rhoA * self.Cdi * self.Ai * vNormDiff * (self.vA - self._agent_velocity)

        fTot = fProp + fDrag

        # step with forward euler
        self._agent_velocity = self._agent_velocity + self.dt * fTot / self.mi
        self._agent_position = self._agent_position + self.dt * self._agent_velocity
        self.counter += 1

        #####################################################################
        # check for crashes, lost agents, targets achieved, etc.
        # nA = 15  # |   none  | SCALAR - Number of initial agents
        # nO = 25  # |   none  | SCALAR - Number of obstacles
        # nT = 100  # |   none  | SCALAR - Number of initial targets
        #####################################################################

        

        # check each agent (agent -> nA)
        for j in range(len(self._agent_position[:, 0])):
            self.atDiff[j, :, :] = (self._target_position - self._agent_position[j])                                 # (nA x nT x 3)
            self.aaDiff[j, :, :] = (self._agent_position - self._agent_position[j])                                 # (nA x nA x 3)
            self.aoDiff[j, :, :] = (self._obstacle_position - self._agent_position[j])                               # (nA x nO x 3)
            self.aaDiff[j, j, :] = np.nan

            self.atDist[j, :] = np.linalg.norm(self._agent_position[j] - self._target_position, ord=2, axis=1)       # (nT x 3) --norm--> (nT x 1)  ∈ (nA, nT)
            self.aaDist[j, :] = np.linalg.norm(self._agent_position[j] - self._agent_position, ord=2, axis=1)       # (nA x 3)  --norm--> (nA x 1) ∈ (nA, nA)
            self.aoDist[j, :] = np.linalg.norm(self._agent_position[j] - self._obstacle_position, ord=2, axis=1)     # (nO x 3)  --norm--> (nO x 1) ∈ (nA, nO)

            # don't count distances of agents with themselves
            self.aaDist[j, j] = np.nan

        # check if agents are in the range of the targets, obstacles, agents, our boundaries
        atHit = np.where(self.atDist < self.agent_sight)                # return ij indices of m-t combos
        aoHit = np.where(self.aoDist < self.crash_range)                # return ij indices of m-o combos
        aaHit = np.where(self.aaDist < self.crash_range)                # return ij indices of m-m combos

        # check for lost agents
        xLost = np.where(np.abs(self._agent_position[:, 0]) > self.xMax)
        yLost = np.where(np.abs(self._agent_position[:, 1]) > self.yMax)
        zLost = np.where(np.abs(self._agent_position[:, 2]) > self.zMax)

        # return i indices of lost drones
        aLost = np.unique(np.hstack([xLost[0], yLost[0], zLost[0]]))

        # index all targets mapped
        target_mapped = np.unique(atHit[1])                                                 # returns indices of m-t contact (j columns are targets)

        # index all agents crashed (from agent, obstacle and lost
        mCrash = np.unique(np.hstack([aaHit[0], aoHit[0], aLost]))

        # remove crashed agents
        self._target_position = np.delete(self._target_position, (target_mapped), axis=0)  # delete x,y,z (row) of _target_position
        self._agent_position = np.delete(self._agent_position, (mCrash), axis=0)
        self._agent_velocity = np.delete(self._agent_velocity, (mCrash), axis=0)
        self.nT -= len(target_mapped)
        self.nA -= len(mCrash)

        self.atDist = np.delete(self.atDist, (mCrash), axis=0)     # remove rows of agents matching to targets
        self.atDiff = np.delete(self.atDiff, (mCrash), axis=0)

        self.aaDist = np.delete(self.aaDist, (mCrash), axis=0)     # remove rows of agents matching to agents
        self.aaDist = np.delete(self.aaDist, (mCrash), axis=1)     # remove columns of agents matching to agents
        self.aaDiff = np.delete(self.aaDiff, (mCrash), axis=0)
        self.aaDiff = np.delete(self.aaDiff, (mCrash), axis=1)

        self.aoDist = np.delete(self.aoDist, (mCrash), axis=0)     # remove rows of agents matching with obstacles
        self.aoDiff = np.delete(self.aoDiff, (mCrash), axis=0)


        # remove mapped targets
        self._target_position = np.delete(self._target_position, (target_mapped), axis=0)             # remove targets that have been mapped
        self.atDist = np.delete(self.atDist, (target_mapped), axis=1)                                 # remove columns of targets interacted with agents
        self.atDiff = np.delete(self.atDiff, (target_mapped), axis=1)

        # compute cost
        Mstar = self.nT / self.nT0
        Tstar = (self.counter * self.dt) / self.tf
        Lstar = ((self.nA0 - self.nA) / self.nA0)
        PI = self.w1 * Mstar + self.w2 * Tstar + self.w3 * Lstar

        # if all agents are lost, crashed, or eliminated, stop the simulation
        if self._target_position.size == 0 or self._agent_position.size == 0 or self.counter == self.total_steps:
            terminated = True
        else: 
            terminated = False


        reward = 100 / PI

        observation = self._get_obs()

        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_size, self.window_size))
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()

    #     canvas = pygame.Surface((self.window_size, self.window_size))
    #     canvas.fill((255, 255, 255))
    #     pix_square_size = (
    #         self.window_size / self.size
    #     )  # The size of a single grid square in pixels

    #     # First we draw the target
    #     pygame.draw.rect(
    #         canvas,
    #         (255, 0, 0),
    #         pygame.Rect(
    #             pix_square_size * self._target_location,
    #             (pix_square_size, pix_square_size),
    #         ),
    #     )
    #     # Now we draw the agent
    #     pygame.draw.circle(
    #         canvas,
    #         (0, 0, 255),
    #         (self._agent_location + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #     )

    #     # Finally, add some gridlines
    #     for x in range(self.size + 1):
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (self.window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, self.window_size),
    #             width=3,
    #         )

    #     if self.render_mode == "human":
    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()

    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:  # rgb_array
    #         return np.transpose(
    #             np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
    #         )

    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()
