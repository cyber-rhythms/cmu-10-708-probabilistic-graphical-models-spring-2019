## Directory contents.

This directory contains 12 `.npz` files generated after running the `metropolis.py` script.

Each `.npz` file contains results from one particular run of the Metropolis algorithm for a particular combination of the MCMC parameters. Each particular combination of MCMC
parameters indexes the filename of the `.npz` file. These MCMC parameters are the isotropic Gaussian proposal standard deviation, `sigma`, and the thinning parameter `t` used
for thinning autocorrelated samples. 

## Loading each .npz file.

Each `.npz` file is a dictionary-like object, and can be accessed for further analysis in say a Jupyter notebook.

- Load the `.npz` file by assigning a name to the output of `np.load()` e.g. `npz_file = np.load(npz_filepath, allow_pickle=True)`
- Where `npz_filepath` is the path to the appropriate `.npz` file you wish to scrutinise.
- You can now reference this object as if it were a dictionary containing 6 key-value pairs.
- Each key is the name of the Numpy data structure, and each value is the Numpy data structure itself. Both were assigned in `metropolis.py`.

## Description of each Numpy data structure in the .npz files.

A description of each data-structure in a typical `.npz` file is as follows:

1. `npzfile['eta_fixed']`

A shape-(4,) ndarray. Hyperparameter initialisations.

2. `npzfile['theta_burn_in_init']`

A shape-(41,) ndarray. Parameter initialisations to initialise the MCMC chain.

3. `npzfile['theta_burn_in']`

A shape-(5001, 41) ndarray. 5000 shape-(1, 41) burn-in iterations of the parameters in the MCMC chain, including initialisation.

4. `npzfile['MCMC_samples']` - 

5. `npzfile['theta_posterior_samples']`

6. `npzfile['acceptance_rate]`






