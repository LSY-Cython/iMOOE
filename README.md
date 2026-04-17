# Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning [ICLR 2026]
This repository is the official PyTorch implementation of iMOOE, which is built upon the defined two-fold PDE invariance principle and capable of achieving zero-shot OOD generalization for PDE dynamics forecasting.

<p float="center">
  <img src="./assets/framework.png" width="800"/>
</p>

# 1. PDE Data Generation
Run `dr.py` and `ns.py` in `datasets/` folder to simulate multi-environment PDE trajectories of Diffusion-Reaction and Navier-Stokes equations, where hyperparameter `num_env_train, num_env_test` determine the number of training and ID/OOD testing environments, `n_data_per_env_train, n_data_per_env_test` determine the number of PDE trajectories under each physical environment.
