## Inferring Wikipedia topics in a latent Dirichlet allocation (LDA) model augmented with a hidden Markov model (HMM) for sequential topic transitions, using variational expectation-maximisation.

This directory contains the materials produced when completing "Q2: Variational inference" of the assignment
"10-708 PGM (Spring 2010): Homework 2 v1.1", whose rubric can be found [here](https://github.com/cyber-rhythms/cmu-10-708-probabilistic-graphical-models-spring-2019/blob/master/homework-assignments/hw-2/hw-2-v1.1.pdf)

The LDA-HMM model specification, mathematical derivations of the co-ordinate ascent variational EM updates, and pseudocode can be found in my write-up
in my write-up of the assignment in the Jupyter notebook [here].

The assignment adapts the original LDA model of Blei, Ng and Jordan (2003) by additionally using a hidden Markov model to account for word-order
and hence sequentiality of topic transitions. This is then applied to the Wikipedia dataset.

## Directory contents.

The contents of this directory are as follows:

1. `environment.yml` - for users wishing to run the code in a conda environment.
2. `requiremnts.txt` - for users wishing to run the code in a Python virtual environment.
3. `/scripts` 

## Getting started.

## Running the code.

## References.

[David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal of machine
Learning research, 3(Jan):993-1022, 2003.](https://jmlr.org/papers/volume3/blei03a/blei03a.pdf)

 