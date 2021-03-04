## Directory contents.

This directory contains 12 `.npz` files generated after running the `metropolis.py` script.

Each `.npz` file contains results from one particular run of the Metropolis algorithm for a particular combination of the MCMC parameters. Each particular combination of MCMC
parameters indexes the filename of the `.npz` file. These MCMC parameters are the isotropic Gaussian proposal standard deviation, `sigma`, and the thinning parameter `t` used
for thinning autocorrelated samples. 

Each `.npz` file is accompanied by a corresponding similarly indexed log `.txt` file in `/logs`.

## Loading each .npz file.

Each `.npz` file is a dictionary-like object, and can be accessed for further analysis in say a Jupyter notebook.

- Load the `.npz` file by assigning a name to the output of `np.load()` e.g. `npz_file = np.load(npz_filepath, allow_pickle=True)`
- Where `npz_filepath` is the path to the appropriate `.npz` file you wish to scrutinise.
- You can now reference this object as if it were a dictionary containing 6 key-value pairs.
- Each key is the name of the Numpy data structure, and each value is the Numpy data structure itself. Both were assigned in `metropolis.py`.

## Description of each Numpy data structure in the .npz files.

A description of each data-structure in a typical `.npz` file is as follows:

1. `npzfile['eta_fixed'] - A shape-(4,) ndarray.` Hyperparameter initialisations. That is, initialisations of the means and variances of common priors on team-specific attacking and defending effects (parameters). Comprising 4 hyperparameter initialisations in total.

2. `npzfile['theta_burn_in_init'] - A shape-(41,) ndarray.` Parameter initialisations to initialise the MCMC chain. That is, initialisations of the global fixed effect home, and the team-specific attacking and defending effects; and in the sense that it is passed as the mean of the isotropic Gaussian proposal distribution from which to draw a sample. Comprises 41 parameters in total. Each parameter is initialised at 0 in accordance with the requirements of the assignment. 

3. `npzfile['theta_burn_in'] - A shape-(5001, 41) ndarray.` Burn-in iterations of the MCMC chain. That is, an initialisation of the shape-(41,) parameters, and then 5000 further iterations.

4. `npzfile['MCMC_samples'] - A shape-(5000t + 1, 41) ndarray.` - Iterations of the MCMC chain. Contains an initialisation of the `shape-(41,)` parameters, taken from the last iteration of the burn-in period. Followed by `5000t` iterations of the MCMC chain, where `t` is the thinning parameter. These are autocorrelated samples of the parameters.

5. `npzfile['theta_posterior_samples'] - A shape(5000, 41) ndarray.` - 5000 samples of the parameters, after thinning. If it is appropriately determined that the MCMC chain has converged to a stationary distribution which is the posterior of interest, then this constitutes 5000 i.i.d samples from the posterior.

6. `npzfile['acceptance_rate] - float` - The empirical acceptance rate for this run of the Metropolis algorithm. Computed using the number of unique samples divided by the total number of iterations in 4.






