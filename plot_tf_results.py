#%% Imports
import os
import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

#%% Inputs 
log_dir = "runs"
print(os.path.join(log_dir,"*"))
path2eventsFolders = glob.glob(os.path.join(log_dir,"*"))
path2eventsFolder = path2eventsFolders[0]
print(path2eventsFolder)

#%% Read in the results 
path2eventsFile = os.path.join(path2eventsFolder, "events.out.*")
eventsFile      = glob.glob(path2eventsFile)[0]

def get_tf_results(file, tagName):
    """
        requires tensorflow==2.10.0
    """
    output = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == tagName:
                output.append(v.simple_value)
    return np.array(output)

collision_rate = get_tf_results(eventsFile, 'results/collision')
avg_reward_per_episode = get_tf_results(eventsFile, 'results/avg_reward')

#%% Plot results 

# Number of episodes 
episodes = np.arange(len(collision_rate))

# Create a figure for the actor loss 
plt.figure(figsize=(8,8))
plt.plot(episodes, collision_rate,'r')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Collusion Toggle', fontsize=20)
plt.show()

print("Collision Rate: ")
print(np.sum(collision_rate))
print("Number of targets found: ")
num_episodes = len(collision_rate)
print(num_episodes - np.sum(collision_rate))

# Create a figure for the average reward  
plt.figure(figsize=(8,8))
plt.plot(episodes, avg_reward_per_episode,'b')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Average Reward', fontsize=20)
plt.show()
