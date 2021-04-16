import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# Below are two helper functions used in the script.

def create_dicts(vocab, tagset):
    """Creates POS tag and vocabulary dictionaries.

    Given ``vocab`` and ``tagset`` arrays, constructs three dictionaries
    that are used to look up a word or POS tag, represented in
    str format, and produces an index, which is used to look
    up appropriate entries of HMM parameter matrices.

    Parameters
    ----------
    vocab: ndarray
        A shape-(V,) object array of V = 12,408 words, represented as
        strings.

    tagset: ndarray
        A shape-(M,) object array of M = 12 POS tags, represented as
        strings.

    Returns
    -------
    out: tuple(dict, dict)
        Tuple containing the following items:

        1. vocab_dict : dict[str, int]
            A dictionary of (word -> word idx) key-value pairs.

        2. tagset_dict: dict[str, int]
            A dictionary of (POS tag -> POS tag idx) key-value pairs.

        3. tagset_dict2 : dict[int, str]
            A dictionary of (POS tag idx -> POS tag) key-value pairs.
    """
    vocab_dict = dict()
    tagset_dict = dict()
    tagset_dict2 = dict()

    for idx in range(V):
        vocab_dict[vocab[idx]] = idx

    for idx in range(M):
        tagset_dict[tagset[idx]] = idx

    for tag in tagset_dict:
        tagset_dict2[tagset_dict[tag]] = tag

    return vocab_dict, tagset_dict, tagset_dict2


def strip_POS_tags(test_set):
    """Pre-process the test-set by stripping out the POS tags.

    ``test_set`` is a shape-(L,) object array consisting of
    L = 783 tagged test sentences, where each test sentence is
    represented as a length T_l[l] list of (word, POS tag) tuples.

    Returns ``test_set_no_tags``, where all POS tags in the tuples
    have been removed.

    Parameters
    ----------
    test_set : ndarray[list]
        A shape-(L,) object array consisting of L = 783 tagged test
        set sentences.

    Returns
    -------
    test_set_no_tags : ndarray[list]
        A shape-(L,) object array consisting of L = 783 untagged test
        set sentences.
    """
    L = test_set.shape[0]
    test_set_no_tags = np.empty((L,), dtype=object)
    T_l = np.array([int(len(test_set[l])) for l in range(L)])

    for l in range(L):
        test_set_no_tags[l] = [test_set[l][t][0] for t in range(T_l[l])]

    return test_set_no_tags


# The core script functionality is in the functions below:

def init_distri(train_set, tagset_dict):
    """Maximum likelihood estimator of the HMM initial distribution.

    Uses ``train_set`` to compute maximum likelihood estimates of
    the hidden Markov model initial distribution parameters, ``pi``.

    ``train_set`` is a shape-(N,) object array consisting of N = 3131
    sentences. Each sentence is represented as a variable length list
    of (word, POS tag) tuples.

    See 'HW2 10-708 write-up.ipynb' for mathematical derivations.

    Parameters
    ----------
    train_set : ndarray
        A shape-(N,) object array consisting of N = 3131 tagged training
        sentences.

    Returns
    -------
    out : tuple(ndarray, ndarray)
        Tuple containing the following items:

        1. pi : ndarray
            A shape-(M,) float array - Maximum likelihood estimate of the
            HMM initial state distribution over M = 12 POS tags.

        2. init_counts : ndarray
            A shape-(M,) float array - The number of occurrences of each of
            M = 12 POS tags in the 1st position of all training sentences.
    """
    tag_idx = 0
    init_counts = np.zeros(M)

    for n in range(N):
        tag_idx = tagset_dict[train_set[n][0][1]]
        init_counts[tag_idx] += 1

    pi = init_counts / init_counts.sum()

    return pi, init_counts


def transition_mat(train_set, tagset_dict):
    """Maximum likelihood estimator of the HMM transition matrix.

    Uses ``train_set`` to compute maximum likelihood estimates of the
    hidden Markov model state transition parameter matrix, ``A``.

    See 'HW2 10-708 write-up.ipynb' for mathematical derivations.

    Parameters
    ----------
    train_set : ndarray
        A shape-(N,) object array consisting of N = 3131 tagged training
        sentences.

    Returns
    -------
    out: tuple(ndarray, ndarray, ndarray)
        Tuple containing the following items:

        1. A : ndarray
            A shape-(M, M) float array - Maximum likelihood estimate of the
            HMM POS tag transition matrix over M = 12 POS tags.

        2. POS_pair_transitions : ndarray
            A shape-(M, M) float array - The number of co-occurrences of POS
            tags in the training data.

        3. POS_transtions : ndarray
            A shape-(M,) float array - The number of transitions from each of
            M = 12 POS tags in the training data.
    """
    POS_transitions = np.zeros(M)
    POS_pair_transitions = np.zeros((M, M))

    for n in range(N):
        for t in range(T_n[n] - 1):
            tag_idx_i = tagset_dict[train_set[n][t:t + 2][0][1]]
            tag_idx_j = tagset_dict[train_set[n][t:t + 2][1][1]]
            POS_pair_transitions[tag_idx_i, tag_idx_j] += 1
            POS_transitions[tag_idx_i] += 1

    A = POS_pair_transitions / POS_transitions[:, np.newaxis]

    return A, POS_pair_transitions, POS_transitions


def emission_mat(train_set, tagset_dict, vocab_dict):
    """Maximum likelihood estimator of the HMM emission matrix.

    Uses ``train_set`` to compute maximum likelihood estimates of the
    hidden Markov model emission parameter matrix, ``B``.

    This is not used for inference directly, due to its sparsity.

    See 'HW2 10-708 write-up.ipynb' for mathematical derivations and
    further info on why this is not used.

    Parameters
    ----------
    train_set : ndarray
        A shape-(N,) object array consisting of N = 3131 tagged training
        sentences.

    Returns
    -------
    out : tuple(ndarray, ndarray, ndarray)
        Tuple containing the following items:

        1. B : ndarray
            A shape-(M, V) float array - Maximum likelihood estimate of
            the HMM word-POS tag emission matrix.

        2. word_POS_counts : ndarray
            A shape-(M, V) float array - The number of co-occurrences of
            each POS tag-word combination in the training data.

        3. POS_counts : ndarray
            A shape-(M,) float array - The number of occurrences of each
            of M = 12 POS tags in the training data.
    """
    POS_counts = np.zeros(M)
    word_POS_counts = np.zeros((M, V))

    for n in range(N):
        for t in range(T_n[n]):
            word_idx = vocab_dict[train_set[n][t][0]]
            tag_idx = tagset_dict[train_set[n][t][1]]
            word_POS_counts[tag_idx, word_idx] += 1
            POS_counts[tag_idx] += 1

    B = word_POS_counts / POS_counts[:, np.newaxis]

    return B, word_POS_counts, POS_counts

def sm_emission_mat(word_POS_counts, POS_counts):
    """Smoothed estimator of HMM emission matrix.

    Applies smoothing and addresses the sparsity of the maximum
    likelihood estimate of the HMM emission matrix.

    The smoothing parameter ``lmbda`` is set at 0.01, as per the
    mandate of the HW2 10-708 assignment.

    Parameters
    ----------
    word_POS_counts : ndarray
        A shape-(M, V) float array - The number of co-occurrences of
        each POS tag-word combination in the training data.

    POS_counts : ndarray
        A shape-(M,) float array - The number of occurrences of each
        of M = 12 POS tags in the training data.

    Returns
    -------
    B_smoothed : ndarray
         A shape-(M, V) float array - A smoothed estimate of the HMM
         word-POS tag emission matrix.
    """
    lmbda = 0.01
    B_smoothed = ((word_POS_counts + lmbda) /
                  (((POS_counts + (lmbda * V))[:, np.newaxis])))
    return B_smoothed


def negative_log_likelihood(pi, A, B, data):
    """Compute joint negative log-likelihood of a tagged dataset.

    Evaluates the negative joint log-likelihood on a given dataset,
    at a particular value of the HMM parameters ``pi``, `A` and ``B``.
    This is a joint log-likelihood, evaluated using the dataset HMM
    observations AND states; and not solely the observations.

    See 'HW2 10-708 write-up.ipynb' for mathematical derivations.

    Parameters
    ----------
    pi : ndarray
        A shape-(M,) float array, HMM initial state distribution parameter
        over M = 12 POS tags.

    A : ndarray
        A shape-(M, M) float array - HMM POS tag transition matrix parameter
        over M = 12 POS tags.

    B : ndarray
        A shape-(M, V) float array - HMM POS tag emission matrix parameter.

    data : ndarray
        An object array containing variable length tagged sentences, each
        represented as

    Returns
    -------
    negative_log_likelihood : float
        The negative joint log-likelihood of the tagged data set.

    """

    N = data.shape[0]
    T_n = np.array([int(len(data[i])) for i in range(N)])
    log_likelihood = 0

    for n in range(N):
        tag_idx = tagset_dict[data[n][0][1]]
        log_likelihood += np.log(pi[tag_idx])
        # print(np.log(pi[tag_idx]))

        for t in range(T_n[n] - 1):
            tag_idx_j = tagset_dict[data[n][t:t + 2][0][1]]
            tag_idx_k = tagset_dict[data[n][t:t + 2][1][1]]
            # if t == 0:
            # print(np.log(A[tag_idx_j, tag_idx_k]))
            log_likelihood += np.log(A[tag_idx_j, tag_idx_k])

        for t in range(T_n[n]):
            word_idx = vocab_dict[data[n][t][0]]
            tag_idx = tagset_dict[data[n][t][1]]
            # if t == 0:
            # print(np.log(B[tag_idx, word_idx]))
            log_likelihood += np.log(B[tag_idx, word_idx])

    negative_log_likelihood = -(log_likelihood)

    return negative_log_likelihood


def viterbi(test_set, pi, A, B, vocab_dict, tagset_dict2):
    """Viterbi decoding for inference of POS tags.

    Given hidden Markov model parameters estimated from the training data,
    ``pi``, ``A``, and ``B``, implements Viterbi decoding to infer the
    jointly most probable sequence of POS tags for every sentence in the
    pre-processed ``test_set``.

    Returns ``tagged_sentences``, a shape-(L,) object array consisting of
    L = 783 variable length tagged sentences. Each tagged sentence is a
    variable length list containing (word, POS tag prediction) tuples.

    See 'HW2 10-708 write-up.ipynb' for mathematical derivations
    and pseudocode.

    Parameters
    ----------
    test_set : ndarray
        A shape-(L,) object array consisting of L = 783 untagged test
        set sentences.
    pi : ndarray
        A shape-(M,) float array, the maximum likelihood estimate of
        the HMM initial state distribution.

    A : ndarray
        A shape-(M, M) float array, the maximum likelihood estimate of
        the HMM state transition matrix.

    B : ndarray
        A shape-(M, V) float array, the smoothed estimate of the HMM
        emission matrix.

    vocab_dict : dict[str, int]
        A dictionary of (word -> word idx) key-value pairs.

    tagset_dict2 : dict[int, str]
        A dictionary of (POS tag idx -> POS tag) key-value pairs.

    Returns
    -------
    tagged_sentences : ndarray
        A shape-(L, ) object array consisting of L = 783 test sentences,
        with POS tag predictions.
    """
    L = test_set.shape[0]
    T_l = np.array([int(len(test_set[l])) for l in range(L)])
    T_max = T_l.max()
    M = tagset.shape[0]
    V = vocab.shape[0]

    # Construct placeholders.
    log_delta = np.full((L, T_max, M), np.nan)
    alpha = np.full((L, T_max, M), -1, dtype=int)
    Y = np.full((L, T_max), -1, dtype=int)
    log_p = np.zeros(L)
    tagged_sentences = np.empty((L,), dtype=object)

    # Initialisation for t = 1.
    for l in range(L):
        word_idx = vocab_dict[test_set[l][0]]
        log_delta[l, 0, :] = np.log(pi) + np.log(B[:, word_idx])
        alpha[l, 0, :] = np.zeros(M)

    # Forward recursion for t = 2, 3,..., T.
    for l in range(L):
        for t in range(1, T_l[l]):
            word_idx = vocab_dict[test_set[l][t]]
            for m in range(M):
                log_delta[l, t, m] = (np.max((log_delta[l, t - 1, :] + np.log(A[:, m])))
                                      + np.log(B[m, word_idx]))
                alpha[l, t, m] = np.argmax((log_delta[l, t - 1, :] + np.log(A[:, m])))

    # Termination for t = T.
    for l in range(L):
        log_p[l] = np.max(log_delta[l, T_l[l] - 1, :])
        Y[l, T_l[l] - 1] = np.argmax(log_delta[l, T_l[l] - 1, :])

    # Path backtracking for t = T - 1,..., 1.
    for l in range(L):
        for t in range(T_l[l] - 2, -1, -1):
            Y[l, t] = np.argmax(log_delta[l, t, :] + np.log(A[:, Y[l, t + 1]]))

    # Convert the predicted POS tag indices into tagged test sentences.
    for l in range(L):
        tagged_sentences[l] = [(test_set[l][t], tagset_dict2[Y[l, t]]) for t in range(T_l[l])]

    return tagged_sentences


def per_word_accuracy(test_set, tagged_sentences):
    """Compute per-word accuracy of predictions.

    Compares the true labels in the unprocessed ``test_set``,
    with that of the predictions in ``tagged_sentences``, and
    returns the proportion of correct predictions.

    Parameters
    ----------
    test_set : ndarray
        A shape-(L,) object array containing L = 783 tagged test
        sentences.

    tagged_sentences : ndarray
        A shape-(L,) object array containing L = 783 test sentences,
        with predicted POS tags on each word.

    Returns
    -------
    per_word_accuracy : float
        The proportion of correct predictions.
    """
    L = test_set.shape[0]
    T_l = np.array([int(len(test_set[l])) for l in range(L)])
    correct = 0

    for l in range(L):
        for t in range(T_l[l]):
            if tagged_sentences[l][t][1] == test_set[l][t][1]:
                correct += 1
            else:
                pass

    per_word_accuracy = correct / T_l.sum()

    return per_word_accuracy


def confusion_matrix(test_set, tagged_sentences, tagset_dict):
    """Compute the confusion matrix.

    Uses unprocessed ``test_set`` and ``tagged_sentences`` to compute
    a shape-(M, M) confusion matrix for M = 12 POS tags.

    Actual POS tag categories in ``test_set`` are listed along axis-0.
    Predicted POS tag categories in ``tagged_sentences`` are listed
    along axis-1.

    No normalisation along either actual or predictions, or entire matrix,
    is applied.

    Parameters
    ----------
    test_set : ndarray[list]
        A shape-(L,) object array consisting of L = 783 tagged test
        set sentences.

    tagged_sentences : ndarray[list]
        A shape-(L,) object array consisting of L = 783 test sentences,
        with predicted POS tags on each word.

    tagset_dict: dict[str, int]
        A dictionary of (POS tag -> POS tag idx) key-value pairs.

    Returns
    -------
    confusion_matrix : ndarray
        A shape-(M, M) int array - the confusion matrix.
    """
    L = test_set.shape[0]
    T_l = np.array([int(len(test_set[l])) for l in range(L)])
    M = len(tagset_dict)
    confusion_matrix = np.zeros((M, M))

    for l in range(L):
        for t in range(T_l[l]):
            actual_tag_idx = tagset_dict[test_set[l][t][1]]
            predicted_tag_idx = tagset_dict[tagged_sentences[l][t][1]]
            confusion_matrix[actual_tag_idx, predicted_tag_idx] += 1

    return confusion_matrix


# The following are further helper functions.

def generate_HMM_confusion_matrix(png_filepath, confusion_matrix, tagset_dict):
    """Generate the confusion matrix as a seaborn heatmap and save.

    Saves the ``confusion_matrix`` as a .png visualisation to
    ``png_filepath`` location.

    Parameters
    ----------
    png_filepath : str
        Path of the new .png visualisation file for saving.

    confusion_matrix : ndarray
        A shape-(M, M) int array, the confusion matrix.

    tagset_dict: dict[str, int]
        A dictionary of (POS tag -> POS tag idx) key-value pairs.
    """
    df_index = df_columns = [tag for tag in tagset_dict.keys()]
    df_cm = pd.DataFrame(confusion_matrix, index=df_index,
                         columns=df_columns, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')

    num_pairs = int(np.sum(confusion_matrix))

    # Annotations
    ax.set_xlabel("Predicted POS tags.")
    ax.set_ylabel("Actual POS tags.")
    ax.set_title("Confusion matrix between POS tags inferred by Viterbi decoding"
                 + "\n"
                 + "in a hidden Markov model and POS tags "
                 + "in the test data set."
                 + "\n"
                 + "\n"
                 + "Total number of actual-predicted POS tag pairs: {}".format(num_pairs))
    # Save figure to specified filepath.
    fig.savefig(png_filepath)


def generate_transition_matrix(png_filepath, truncated_trans_mat, tagset_dict):
    """Generate the transition matrix as a seaborn heatmap and save.

    Saves the ``truncated_trans_mat`` as a .png visualisation to
    ``png_filepath`` location.

    Parameters
    ----------
    png_filepath : str
        Path of the new .png visualisation file for saving.

    truncated_trans_mat : ndarray
        A shape-(M, M) float array, maximum likelihood estimate of
        the HMM POS tag transition matrix, rounded to 3 decimal places.

    tagset_dict: dict[str, int]
        A dictionary of (POS tag -> POS tag idx) key-value pairs.
    """
    df_index = df_columns = [tag for tag in tagset_dict.keys()]
    df_tm = pd.DataFrame(truncated_trans_mat, index=df_index,
                         columns=df_columns, dtype=float)

    # Make plot and call axes level function heatmap from seaborn.
    fig, ax = plt.subplots(figsize=(10, 9))
    ax = sns.heatmap(df_tm, annot=True, cmap='Blues', fmt='.3f')

    # Annotations
    ax.set_title("Maximum likelihood estimate of the hidden Markov model POS tag state"
                 + "\n"
                 + "transition probability matrix, rounded to 3 decimal places.")

    ax.set_xlabel("Current POS tag, $y_{t}$")
    ax.set_ylabel("Previous POS tag, $y_{t-1}$")

    # Save figure to specified filepath.
    fig.savefig(png_filepath)

def save_results(npz_filepath):
    """Save script results as an .npz file.

    Saves main results of the script in a Numpy accessible format
    to ``npz_filepath`` location.

    Parameters
    ----------
    npz_filepath : str
        Path of the new .npz results file for saving.
    """
    names = ['pi', 'A', 'B', 'B_smoothed',
             'joint NLL (train)',
             'joint NLL (test)',
             'HMM Viterbi tagged sentences',
             'per word accuracy',
             'confusion matrix']

    arrays = [pi, A, B, B_smoothed,
              joint_NLL_train,
              joint_NLL_test,
              tagged_sentences,
              per_word_accuracy,
              confusion_matrix]

    kwargs = dict()
    for name, array in zip(names, arrays):
        kwargs[name] = array

    np.savez(npz_filepath, **kwargs)

if __name__ == '__main__':

    # Get current working directory path.
    cwd = os.getcwd()

    # Create a '/results' subdirectory.
    results_dir_path = os.path.join(cwd, 'results')
    isdir = os.path.isdir(results_dir_path)
    if isdir:
        pass
    else:
        os.mkdir(results_dir_path)

    # Redirect stdout to a .txt file in '/results/ subdirectory.
    txt_filepath = os.path.join(results_dir_path, 'pos-hmm-selected-results.txt')
    sys.stdout = open(txt_filepath, 'w')

    # Load the data from the subdirectory '/pos-data'
    # As per instructions in repository, this subdirectory should be
    # placed within the directory where this script is located.
    train_set = np.load('pos-data/train_set.npy', allow_pickle=True)
    test_set = np.load('pos-data/test_set.npy', allow_pickle=True)
    vocab = np.loadtxt('pos-data/vocab.txt', dtype=object)
    tagset = np.loadtxt('pos-data/tagset.txt', dtype=object)

    # The vocabulary is missing the character '#', which is in the training set.
    # The following step remedies this.
    vocab = np.append(vocab, '#')

    # Define relevant dimensions to be used in the script.
    N = train_set.shape[0]
    M = tagset.shape[0]
    V = vocab.shape[0]

    # Construct an array containing variable training set sentence lengths.
    T_n = np.array([int(len(train_set[i])) for i in range(N)])

    # Pre-process the test set and bin the POS tags.
    test_set_no_tags = strip_POS_tags(test_set)

    # Create dictionaries.
    vocab_dict, tagset_dict, tagset_dict2 = create_dicts(vocab, tagset)

    # Supervised learning/maximum likelihood estimation of the HMM
    # initial distribution, state transition matrix, emission matrix.
    pi, _ = init_distri(train_set, tagset_dict)
    A, _, _ = transition_mat(train_set, tagset_dict)
    B, word_POS_counts, POS_counts = emission_mat(train_set, tagset_dict, vocab_dict)

    # Smoothed maximum likelihood estimates of the emission probability matrix.
    B_smoothed = sm_emission_mat(word_POS_counts, POS_counts)

    # Compute negative joint log-likelihood for the training and test data sets.
    joint_NLL_train = negative_log_likelihood(pi, A, B_smoothed, train_set)
    joint_NLL_test = negative_log_likelihood(pi, A, B_smoothed, test_set)

    # Viterbi decoding.
    tagged_sentences = viterbi(test_set_no_tags,
                               pi, A, B_smoothed,
                               vocab_dict, tagset_dict2)

    # Compute per-word accuracy.
    per_word_accuracy = per_word_accuracy(test_set, tagged_sentences)

    # Compute confusion matrix as a simple Numpy array.
    confusion_matrix = confusion_matrix(test_set, tagged_sentences, tagset_dict)

    # Create a '/figures' subdirectory to store .png visualisations.
    figures_dir_path = os.path.join(cwd, 'figures')
    isdir = os.path.isdir(figures_dir_path)
    if isdir:
        pass
    else:
        os.mkdir(figures_dir_path)

    # Generate and store .png visualisation of confusion matrix.
    png_filepath = os.path.join(figures_dir_path, 'pos-hmm-confusion-matrix.png')
    generate_HMM_confusion_matrix(png_filepath, confusion_matrix, tagset_dict)

    # Save results into an .npz file in '/results' subdirectory.
    npz_filepath = os.path.join(results_dir_path, 'pos-hmm-results.npz')
    save_results(npz_filepath)

    # Print the specific results requested in HW2 10-708 Q1.1.2 and Q1.1.3.
    print("The obtained HMM POS tag state transition matrix, A, "
          + "rounded to 3 decimal places, is: ")
    A_truncated = np.round(A, 3)
    print(A_truncated)

    print("The negative joint log-likelihood over the training set is: "
          + "{}".format(joint_NLL_train))


    print("The negative joint log-likelihood over the test set is: "
          + "{}".format(joint_NLL_test))

    print("The per word accuracy, as a percentage rounded to 3 decimal places, is: "
          + "{} %".format(np.round(per_word_accuracy * 100, 3)))

    # Generate and save .png visualisation of truncated maximum likelihood
    # estimate of the HMM POS-tag state transition matrix to '/results' subdirectory.
    png_filepath2 = os.path.join(figures_dir_path, 'pos-hmm-mle-transition-matrix.png')
    generate_transition_matrix(png_filepath2, A_truncated, tagset_dict)













