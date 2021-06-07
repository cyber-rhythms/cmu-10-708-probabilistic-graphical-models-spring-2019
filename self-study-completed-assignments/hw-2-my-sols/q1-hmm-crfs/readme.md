## Using hidden Markov models (HMMs) and linear-chain conditional random fields (CRFs) for parts-of-speech tagging of Wall Street Journal articles in the Penn Treebank corpus.

This directory contains the materials I produced as part of self-study when completing "Q1. Sequential models for POS tagging" of the assignment "10-708 PGM (Spring 2019): Homework 2 v1.1", whose rubric can be found [here](https://github.com/cyber-rhythms/cmu-10-708-probabilistic-graphical-models-spring-2019/blob/master/homework-assignments/hw-2/hw-2-v1.1.pdf).

*Both scripts in this directory are used for implementing training and inference of the most likely sequence of parts-of-speech tags applied to the Wall Street Journal portion of the Penn Treebank corpus.*

*As part of a supervised-learning procedure, the script `pos-hmm.py` uses parts-of-speech tagged training sentences to estimate the parameters of a hidden Markov model via maximum likelihood estimation. The trained discriminative model is then used to predict the most likely sequence of parts-of-speech tags on test data using a Viterbi decoding algorithm.*

*The script `pos-linear-crf.py` uses training data to estimate the parameters on a linear-chain conditional random field, again via maximum likelihood estimation. As a necessary subroutine within the training procedure, computation of unary and pairwise marginal posterior distributions via a junction-tree message passing algorithm is implemented, and parameters are iteratively updated via stochastic gradient descent until convergence. Inference/sequence prediction then proceeds on test data by using the trained generative model together with a Viterbi decoding algorithm.*

The specification of the HMM and linear-chain CRF model, mathematical derivations and pseudocode for the training and inference procedures of both models which pertain to this repository are located in my write-up of the assignment in the Jupyter notebook [here].

## Directory contents.

1. `environment.yml` - for users wishing to run the code in a conda environment.
3. `requirements.txt` - for users wishing to run the code in a Python virtual environment. 
4. `pos-hmm.py` - script for training of and inference using a hidden Markov model.
5. `pos-linear-crf.py` - script for training and inference using a linear-chain conditional random field.
6. `/pos-data` - the Wall Street Journal portion of the Penn Treebank corpus. Contains tag set and vocabulary as `.txt` files; and training and test data as `.npy` files.
7. `/figures` - figures generated by the scripts.
8. `/results` - script output.

## Getting started.

In order for the script to run appropriately, use conda (recommended) or Python virtual environments. For those wishing to run the script without recourse to either of these methods, the core packages used in the script are also listed.

### Using conda (recommended).

Download the `environment.yml` file to an appropriate working directory. Create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment using the provided `environment.yml` file. That is, open a terminal or Anaconda3 prompt and run `conda env create -f environment.yml`. This will create an environment named `hmm-crfs`. Now activate this environment by running `conda activate hmm-crfs`. You can now run the script from the terminal.

### Using a Python (3.8.8) virtual environment.

Download the `requirements.txt` file to an appropriate working directory. Open a terminal, activate the Python virtual environment and run `pip install -r requirements.txt`. The relevant packages used to run the script should be installed. You can now run the terminal.

### Core packages used in the script.

For those wishing to run the script using other means, the core packages used in the script are the following***:

```
matplotlib==3.3.4
scipy==1.5.2
numpy==1.19.2
seaborn==0.11.1
pandas==1.2.4
```

N.B This will have to be amended once PyTorch CRF dependencies are ported from Jupyter.

## Running the code.

1. Place the main scripts (either `pos-hmm-crfs.py` or `pos-linear-crf.py` or both) and the entire data folder `/pos-data` in a working directory of your choice. Please ensure that both the main scripts and the data folder are placed in the same working directory, and that namings are not altered, otherwise the script will not run appropriately.

2. Execute either of the main scripts by running `python pos-hmm.py` or `python pos-linear-crf.py`.

3. Executing either of the main scripts will create 2 new sub-directories in your current working directory. Those sub-directories are `/results` and `/figures`. The script will output results in the form of `.txt` files and `.npz` files, which will be stored in the `/results` sub-directory; and a number of `.png` visualisations which are stored in `/figures`. Further information can be found in the `readme.md` of the relevant sub-directory.

## Directions for further exploration.

tbc

## References.

[1] [John Lafferty, Andrew McCallum, and Fernando CN Pereira. Conditional random fields: Probabilistic
models for segmenting and labeling sequence data. 2001.](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)

[2] [Slav Petrov, Dipanjan Das, and Ryan McDonald. A universal part-of-speech tagset. arXiv preprint
arXiv:1104.2086, 2011.](https://arxiv.org/pdf/1104.2086.pdf)