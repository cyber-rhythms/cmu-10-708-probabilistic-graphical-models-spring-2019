## Posterior inference in a Bayesian hierarchical model using the Metropolis algorithm: Sports analytics of 2013-2014 Premier League goal data.



This directory contains the materials produced when completing "Q4: Markov Chain Monte Carlo" of the assignment "10-708 PGM (Spring 2019): Homework 2 v1.1", whose rubric can be found [here](https://github.com/cyber-rhythms/cmu-10-708-probabilistic-graphical-models-spring-2019/blob/master/homework-assignments/hw-2/hw-2-v1.1.pdf)

The Bayesian hierarchical model specification, pseudocode, mathematical derivations, and analysis are located in my write-up of the assignment in the Jupyter notebook [here].

The assignment is an adaptation of the Bayesian hierarchical model of Baio and Blangiardo (2010) to sports data for the 2013-2014 Premier League football results, rather than Italian Serie A results.

## Directory contents.

The contents of this directory are as follows:

1. `metropolis.py` - the main script for running the Metropolis algorithm.
2. `premier_league_2013_2014.dat` - 2013-2014 Premier League data. 
3. `PRNG_state.npy` - a pseudo-random number generator (PRNG) state file for reproducible results.
4. `requirements.txt` - to ensure compatibility.
5. `MCMC analysis.ipynb` - a supplementary Jupyter notebook used to generate a posterior histogram and scatter plot of estimated posterior means of the parameters.
6. `/results` - a directory containing the output of the script `metropolis.py` and the Jupyter notebook `MCMC analysis.ipynb`.

## Getting started.

## Running the code from scratch.

1.  Place the script `metropolis.py`, the data `premier_league_2013_2014.dat`, and the PRNG state file `PRNG_state.npy` into any working directory of your choice. However, please ensure that these three files are placed in the *same working directory*, otherwise the script will not execute as intended.

2. Execute the script by opening a terminal with Python installed and running `python metropolis.py`

3. Executing the script `metropolis.py` will create a new directory `/results` in your current working directory (where you have saved the files specified in step 1.). Running this script will generate a log-file `metropolis-log.txt` containing the `stdout` of the script; 12 `.npz` files and their 12 corresponding `.png` traceplots; and finally, `acceptance-rate-table.png`, a table of empirical acceptance rates for each run of the Metropolis algorithm.

4. *After* the script `metropolis.py` has completed, open the Jupyter notebook `MCMC analysis.ipynb` and run each of the code cells. Doing this will cause `MCMC analysis.ipynb` to access the `/results` directory, and unpack `metropolis-results-sigma=0.005-t=50.npz`. It will then generate the required posterior histogram and scatter plot of estimated empirical means of the attacking and defence strength parameters for each of the Premier League teams. These visualisations will be saved as `posterior-density-histogram.png` and `scatter-plot-estimated-posterior-means.png` in the `/results` directory.

## Plots.

## References

[Gianluca Baio and Marta Blangiardo. Bayesian hierarchical model for the prediction of football results.
Journal of Applied Statistics, 37(2):253-264, 2010.](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf)



