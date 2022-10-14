# Soft Actor Critic Algorithm
### We present a PyTorch implementation of the Soft Actor Critic (SAC) Algorithm proposed by [Haarnoja et al.](https://arxiv.org/abs/1812.05905). This particular implementation adjusts the entropy scaling coefficient, $\alpha$, with each iteration (and therefore is not to be confused with the original implementation of SAC ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)) where $\alpha$ is treated as a hyperparameter).

# Requirements and set up
Begin by creating and activating a conda environment in the terminal: 
```
conda env -n sac python=3.7
conda activate sac
```
