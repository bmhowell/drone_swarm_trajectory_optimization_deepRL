"""
prior to running: 
    - source .env/bin/activate
    - export PYTHONPATH="${/usr/bin/python3}:/Users/brianhowell/Desktop/Berkeley/Fall2022/COMPSCI 285/projects/drone_swarm_trajectory_optimization_deepRL/"
    
    RUNNING EXAMPLE ENVIRONMENT
"""

import gym_examples
from gym_examples.wrappers import relative_position
import gym 

env = gym.make('gym_examples/GridWorld-v0')
wrapped_env = relative_position.RelativePosition(env)
print(wrapped_env.reset())     # E.g.  [-3  3], {}


"""
    RUNNING DRONE_SWARM ENVIRONMENT
"""

# import gym_examples
# from gym_examples.wrappers import RelativePosition
# import gym 
print('\nDRONE\n')
env = gym.make('gym_examples/GameOfDrones-v0')
wrapped_env = relative_position.RelativeDronePosition(env)
print(wrapped_env.reset())     # E.g.  [-3  3], {}