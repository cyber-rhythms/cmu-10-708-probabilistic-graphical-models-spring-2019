## Inferring Wikipedia topics in a latent Dirichlet allocation (LDA) model augmented with a hidden Markov model (HMM) for sequential topic transitions, using variational expectation-maximisation.

This directory contains the materials I produced as part of self-study when completing "Q2: Variational inference" of the assignment
"10-708 PGM (Spring 2010): Homework 2 v1.1", whose rubric can be found [here](https://github.com/cyber-rhythms/cmu-10-708-probabilistic-graphical-models-spring-2019/blob/master/homework-assignments/hw-2/hw-2-v1.1.pdf).

*Both scripts in this directory implement a variational expectation-maximisation algorithm on a pre-processed Wikipedia data set to estimate the topic-word distribution matrix (parameter) of an adapted latent Dirichlet allocation model. Where the latent Dirichlet allocation model has been augmented with a hidden Markov model whose transition matrix is used to model sequential latent topic transition probabilities.*

*The variational expectation-maximisation algorithm proceeds in an iterative two-step procedure. Given an initialisation of/fixed current estimates of model parameters and hyperparameters, the variational E-step sequentially updates each variational parameter within a set in turn, whilst holding the remaining variational parameters fixed, continuing until convergence of the evidence lower bound for that specific model (hyper)-parametrisation. The variational M-step then updates the model parameters, given estimates of the variational parameters computed in the E-step. These variational EM update sequences are continued until convergence of the evidence lower bound (ELBO).The difference between the two scripts, `lda-hmm-vem-empirical-bayes.py` and `lda-hmm-vem-hyp-tune.py`, is the treatment of the model hyperparameters.*

*The script `lda-hmm-vem-empirical-bayes.py`, in addition to updating the model parameters to maximise the ELBO in the variational M-step, will additionally update model hyperparameters, resulting in empirical Bayes estimates of the latter if the variational EM update sequences converge.*

*The latter script `lda-hmm-vem-hyp-tune.py` does not update the model hyperparameters, rather, tunes these hyperparameters by fitting multiple models and then selecting the hyperparameter setting (indexing a model) that performs the best on a metric computed on a held-out test-set. In this simplified setting, the metric used is the ELBO.*

The LDA-HMM model specification, mathematical derivations of the co-ordinate ascent variational EM updates, and pseudocode can be found in my write-up
in my write-up of the assignment in the Jupyter notebook [here].

The assignment adapts the original LDA model of Blei, Ng and Jordan (2003) by additionally using a hidden Markov model to account for word-order
and hence sequentiality of topic transitions. This is then applied to the Wikipedia dataset.

## Directory contents.

The contents of this directory are as follows:

1. `environment.yml` - for users wishing to run the code in a conda environment.
2. `requirements.txt` - for users wishing to run the code in a Python virtual environment.
3. `lda-hmm-vem-empirical-bayes.py` - script that runs variational EM with empirical Bayes estimation of the hyperparameters.
4. `lda-hmm-vem-hyp-tune.py` - script that runs variational EM without estimating the hyperparameters, which are instead tuned via a grid-search.
5. `/data` - 
6. `/logs` - 
7. `/results` - 
8. `/figures` - 

## Getting started.

## Running the code.

## References.

[David M Blei, Andrew Y Ng, and Michael I Jordan. Latent dirichlet allocation. Journal of machine
Learning research, 3(Jan):993-1022, 2003.](https://jmlr.org/papers/volume3/blei03a/blei03a.pdf)

 
