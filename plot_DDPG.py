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
print(path2eventsFolders)
path2eventsFolder = path2eventsFolders[1]
print(path2eventsFolder)

#%% Read in the results 
path2eventsFile = os.path.join(path2eventsFolder, "events.out.*")
eventsFile      = glob.glob(path2eventsFile)[0]
print('eventsFile: ', eventsFile)

def get_tf_results(file, tagName):
    """
        requires tensorflow==2.10.0
    """
    x_axis = []
    output = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            # print(v.tag)
            if v.tag == tagName:
                output.append(v.simple_value)
                # Extract the x and y values from the summary
                # print(v.simple_value)
    return np.array(output)

avg_crit_loss = get_tf_results(eventsFile, 'losses/critic_loss')
avg_actor_loss = get_tf_results(eventsFile, 'losses/actor_loss')

#%% Plot results 

x_axis_critic = np.arange(start=0, stop=len(avg_crit_loss)*1000, step=1000) + 10000
x_axis_actor = np.arange(start=0, stop=len(avg_actor_loss)*1000, step=1000) + 10000

# Create a figure for the actor loss 
plt.figure(figsize=(8,8))
plt.plot(x_axis_actor, avg_actor_loss,'r')
plt.xlabel('Number of environment steps', fontsize=20)
plt.ylabel('Average Actor Loss', fontsize=20)
# Save the figure
plt.savefig('actor_loss_DDPG.png')

# Create a figure for the average reward  
plt.figure(figsize=(8,8))
plt.plot(x_axis_critic, avg_crit_loss,'b')
plt.xlabel('Number of environment steps', fontsize=20)
plt.ylabel('Average Critic Loss', fontsize=20)
# Save the figure
plt.savefig('critic_loss_DDPG.png')
