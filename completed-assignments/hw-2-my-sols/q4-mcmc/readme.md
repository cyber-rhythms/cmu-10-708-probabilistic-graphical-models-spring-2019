## Posterior inference in a Bayesian hierarchical model using the Metropolis algorithm: Sports analytics of 2013-2014 Premier League goal data.

This directory contains the materials produced when completing "Q4: Markov Chain Monte Carlo" of the assignment "10-708 PGM (Spring 2019): Homework 2 v1.1". 

The assignment rubric is located [here](https://github.com/cyber-rhythms/cmu-10-708-probabilistic-graphical-models-spring-2019/blob/master/homework-assignments/hw-2/hw-2-v1.1.pdf).

Pseudocode, mathematical derivations, and analysis are located in the Jupyter notebook [here].

The assignment is an adaptation of the Bayesian hierarchical model of Baio and Blangiardo (2010) to sports data for the 2013-2014 Premier League football results, rather than Italian Serie A results.

The contents of this directory are as follows:

1. `metropolis.py` - the main script to run the Metropolis algorithm.
2. `premier_league_2013_2014.dat` - 2013-2014 Premier Leagure data. 
3. `PRNG_state.npy` - a pseudo-random number generator (PRNG) state file.
4. `requirements.txt` - 
5. `MCMC analysis.ipynb` - a supplementary Jupyter notebook used to generate a histogram and scatter plot.
6. `/results` - a directory containing the output of `metropolis.py` and `MCMC analysis.ipynb`.

### Getting started.

### Running the code from scratch.

1.  Place the script `metropolis.py`, the data `premier_league_2013_2014.dat`, and the PRNG state file `PRNG_state.npy` into the *same working directory* of your choice.

2. Execute the script by opening a terminal with Python installed and running `python metropolis.py`

3. The script will create a new directory `/results` in your current working directory (where you have saved the files specified in step 1.). It will then save a total of 12 `.npz` files using the 

### Plots.

### References

