## Posterior inference in a Bayesian hierarchical model using the Metropolis algorithm: Sports analytics with 2013-2014 English Premier League football results.

>**The main script in this directory conducts posterior inference of team attack and defence strength (parameters) for 20 English >Premier League football teams in a Bayesian hierarchical model, given a random initialisation of the prior means and variances (hyperparameters) on these attacking and defence strengths, and goals (observed data) from the 2013-2014 season. The posterior inference procedure uses a random walk Metropolis algorithm with a isotropic Gaussian symmetric proposal distribution, which is an instance of the more general class of simulation-based Markov chain Monte Carlo methods.**

This directory contains the materials produced when completing "Q4: Markov Chain Monte Carlo" of the assignment "10-708 PGM (Spring 2019): Homework 2 v1.1", whose rubric can be found [here](https://github.com/cyber-rhythms/cmu-10-708-probabilistic-graphical-models-spring-2019/blob/master/homework-assignments/hw-2/hw-2-v1.1.pdf).

The Bayesian hierarchical model specification, pseudocode, mathematical derivations, and analysis whiuch pertain to this repository are located in my write-up of the assignment in the Jupyter notebook [here].

The assignment is an adaptation of the Bayesian hierarchical model of Baio and Blangiardo (2010) to data for the 2013-2014 Premier League football results, rather than Italian Serie A results.

## Directory contents.

The contents of this directory are as follows:

1. `environment.yml` - for users wishing to run the code in a conda environment.
2. `requirements.txt` - for users wishing to run the code using a Python virtual environment.
3. `metropolis.py` - the main script for running the Metropolis algorithm.
4. `premier_league_2013_2014.dat` - 2013-2014 Premier League data. 
5. `PRNG_state.npy` - a pseudo-random number generator (PRNG) state file for use with Numpy, ensures reproducibility of the results.
6. `MCMC-suppl-visualisations.ipynb` - a supplementary Jupyter notebook used to generate a posterior histogram and scatter plot of estimated posterior means of the parameters.
7. `/results` - sub-directory containing `.npz` files output from the the main script. 
8. `/logs` - a sub-directory containing `.txt` files containing console messages output from the main script.
9. `/figures` - a sub-directory containing figures generated as per requirements of the assignment.

## Getting started.

In order for the script to run appropriately, use conda (recommended) or Python virtual environments. The packages used in the script are specfied for those wishing to run the script using other meanns.

### Using conda.

Download the `environment.yml` file to an appropriate working directory. Create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment using the provided `environment.yml` file. That is, open a terminal or Anaconda3 prompt and run `conda env create -f environment.yml`. Then activate the environment by running `conda activate hw2-10-708`. You can now run the script from this terminal.

### Using a Python (3.8.8) virtual environment.

Download the `requirements.txt` file to an appropriate working directory. Open a terminal, activate the Python virtual environment and run `pip install -r requirements.txt`. The relevant packages used to run the script should be installed. You can now run the script from this terminal.

### Packages used in the script.

For those wishing to run the script using other means, the core packages used in the script are the following:

```
matplotlib==3.3.4
scipy==1.5.2 
numpy==1.19.2
```

## Running the code from scratch.

1.  Place the main script `metropolis.py`, the data `premier_league_2013_2014.dat`, and the PRNG state file `PRNG_state.npy` into a working directory of your choice. It is assumed that this working directory is the same as where you placed the `environment.yml` or `requirements.txt` file. Please ensure that the former three files are placed in the *same working directory*, otherwise the script will not execute as intended.

2. Execute the main script by running `python metropolis.py` in your terminal.

3. Executing the main script `metropolis.py` will create 3 new sub-directories in your current working directory (where you have saved the files specified in step 1). Those sub-directories are `/logs`, `/results`, `/figures`. Furthermore, the script will produce `.txt` log files, which are saved in `/logs`; `.npz` results files, which are saved in `/results`, and a number of `.png` visualisations, which are saved in `/figures`. Further information on these can be found in the `readme.md` of the relevant sub-directories.

5. *After* the mainscript `metropolis.py` has terminated, place the Jupyter notebook `MCMC-suppl-visualisations.ipynb` in your current working directory, then open it and run each of the code cells. `MCMC-suppl-visualisations.ipynb` will access the `/results` sub-directory, and unpack `metropolis-results-sigma=0.005-t=50.npz`. It is important to wait until either the main script has terminated (or at least until `metropolis-results-sigma=0.005-t=50.npz` has been generated in `/results`) before running the code cells in the Jupyter notebook. The notebook will then generate the required posterior histogram and scatter plot of estimated empirical means of the attacking and defence strength parameters for each of the Premier League teams. These visualisations will be saved as `posterior-density-histogram.png` and `scatter-plot-estimated-posterior-means.png` in the `/results` directory.

## Directions for further exploration.



## References

[Gianluca Baio and Marta Blangiardo. Bayesian hierarchical model for the prediction of football results.
Journal of Applied Statistics, 37(2):253-264, 2010.](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)



