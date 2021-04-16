## Directory contents.

This directory contains `.npz` and `.txt` output files generated from the main scripts, `pos-hmm.py` and `pos-linear-crfs.py`. Each `.npz` and `.txt` filename is indexed by the name of the script.

The following are instructions on how to load the `.npz` files for debugging, reproducibilty, or analysis; and a description of the contents of each `.npz` file. The `.txt` files contain information that is used in the assignment write-up `HW2 10-708 write-up.pdf`.

## Loading each .npz file.

Each `.npz` file is a dictionary-like object, and can be accessed for further analysis in say a Jupyter notebook.

- Load the `.npz` file by assigning a name to the output of `np.load()` e.g. `npz_file = np.load(npz_filepath, allow_pickle=True)`
- Where `npz_filepath` is the path to the appropriate `.npz` file you wish to scrutinise.
- You can now reference this object as if it were a dictionary containing key-value pairs.
- Each key is the name of the Numpy data structure, and each value is the Numpy data structure itself.

## Description of Numpy data structures in `pos-hmm-results.npz`.

1. `npzfile['pi'] - A shape-(12,) ndarray.` Maximum likelihood estimates of the hidden Markov model (HMM) initial state distribution matrix, corresponding to a discrete Multinoulli distribution over possible POS tags.

2. `npzfile['A'] - A shape-(12, 12) ndarray.` Maximum likelihood estimates of the HMM POS tag discrete state transition matrix. 

3. `npzfile['B'] - A shape-(12, 12408) ndarray.` Maximum likelihood estimates of the HMM word-POS tag discrete emission matrix. Sparse and therefore not used for inference, but included for reference.

4. `npzfile['B_smoothed'] - A shape-(12, 12408) ndarray.` A smoothed estimate of the HMM word-POS tag discrete emission matrix, using pseudocounts.

5. `npzfile['joint NLL (train)'] - float.` The joint negative log-likelihood of the HMM, evaluated using the tagged training sentences as the data, and using the HMM maximum likelihood estimates of the initial distribution, state transition matrix, but smoothed estimates of the emission matrix.

6. `npzfile['joint NLL (test)'] - float.` The joint negative log-likelihood of the HMM, evaluated using the tagged test sentences as the data, and using the HMM maximum likelihood estimates of the initial distribution, state transition matrix, but smoothed estimates of the emission matrix.

7. `npzfile['HMM Viterbi tagged sentences'] - A shape-(783,) ndarray.` The tagged sentences containing POS tag predictions output by the Viterbi algorithm. Consists of 783 variable length lists of tagged sentences, where each sentence consists of a variable number of (word, POS tag prediction) tuples.

8. `npzfile['per word accuracy'] - float.` The per-word accruracy of the POS tag predictions output by Viterbi decoding, compared with the tagged test sentences.

9. `npzfile['confusion matrix'] - A shape-(12, 12) ndarray.` The confusion matrix of actual tagged test-data POS tags and predicted POS tags from Viterbi decoding. Unnormalised, and is computed for a total 20334 actual-predicted POS pairs.





