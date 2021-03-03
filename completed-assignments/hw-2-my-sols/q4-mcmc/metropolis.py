import numpy as np
from scipy.stats import norm, gamma, poisson, multivariate_normal, uniform
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os


def parameter_grid():
    """Creates a grid of MCMC parameter settings to run the Metropolis algorithm.

    The floats contained within `sigmas` and `thinning_params` are values of the following
    MCMC parameter settings:

    i) sigma - the standard deviation of the isotropic Gaussian proposal distribution
    used in the Metropolis algorithm.

    ii) thinning_param - the thinning parameter, a means of adjusting for autocorrelation of
    samples generated from the MCMC chain. A posterior* sample is collected from the MCMC
    chain every `thinning_param` time steps, and the rest are eventually discarded.

    *If it is indeed assessed via diagnostics that the stationary distribution to which
    the MCMC chain has converged is the posterior distribution of interest.

    See "HW2 10-708 write-up.ipynb" for further info.

    Returns
    -------
    sigmas : List[float]
        A list of floats containing isotropic Gaussian proposal distr. standard deviations.

    thinning_params : List[float]
        A list of floats containing values of the thinning parameter.

    num_settings: int
        The number of combinations of MCMC parameter settings to run the Metropolis algorithm.
    """
    sigmas= [0.005, 0.05, 0.5]
    thinning_params = [1, 5, 20, 50]
    num_settings = len(sigmas) * len(thinning_params)

    return sigmas, thinning_params, num_settings

def compute_log_density(theta, eta, data):
    """Compute the log joint density of the model.

    Given observed `data`, model parameters `theta`, and hyperparameters `eta`,
    outputs the log joint density of the model. Used in the computation of the
    acceptance probability (in log-scale) and the implementation of the decision
    rule for accepting/rejecting candidate samples drawn from the proposal
    distribution in the Metropolis algorithm.

    See "HW2 10-708 write-up.ipynb" for further info.

    Parameters
    ----------
    theta : ndarray
        A shape-(41,) ndarray containing the model parameters.

    eta : ndarray
        A shape-(4,) ndarray containing the model hyperparameters.

    data : ndarray
        A shape-(380, 4) int ndarray containing the observed data.

    Returns
    -------
    log_density : float
        A float - the log joint density of the model.
    """
    num_games = data.shape[0]

    mu_att = eta[0]
    mu_def = eta[1]
    tau_att = eta[2]
    tau_def = eta[3]

    home = theta[0]
    atts = theta[1:21]
    defs = theta[21:41]

    home_goals = data[:, 0]
    away_goals = data[:, 1]
    home_idx = data[:, 2]
    away_idx = data[:, 3]

    alpha = 0.1
    beta = 0.1
    tau_0 = 0.0001
    tau_1 = 0.0001

    log_p_mu_att = np.log(norm.pdf(mu_att, loc=0, scale=np.sqrt(tau_1)))
    log_p_mu_def = np.log(norm.pdf(mu_def, loc=0, scale=np.sqrt(tau_1)))
    log_p_tau_att = np.log(gamma.pdf(tau_att, a=alpha, scale=(1 / beta)))
    log_p_tau_def = np.log(gamma.pdf(tau_def, a=alpha, scale=(1 / beta)))

    log_p_home = np.log(norm.pdf(home, loc=0, scale=np.sqrt(tau_0)))

    log_p_home_atts = np.log(norm.pdf(atts[home_idx], loc=mu_att,
                                      scale=np.sqrt(tau_att)))

    log_p_home_defs = np.log(norm.pdf(defs[home_idx], loc=mu_def,
                                      scale=np.sqrt(tau_def)))

    log_p_away_atts = np.log(norm.pdf(atts[away_idx], loc=mu_att,
                                      scale=np.sqrt(tau_att)))

    log_p_away_defs = np.log(norm.pdf(defs[away_idx], loc=mu_def,
                                      scale=np.sqrt(tau_def)))

    theta_0 = np.exp(home + atts[home_idx] - defs[away_idx])
    theta_1 = np.exp(atts[away_idx] - defs[home_idx])

    log_p_home_goals = np.log(poisson.pmf(home_goals, mu=theta_0))
    log_p_away_goals = np.log(poisson.pmf(away_goals, mu=theta_1))

    log_density = (log_p_mu_att + log_p_mu_def + log_p_tau_att + log_p_tau_def + log_p_home
                   + np.sum(log_p_home_atts + log_p_home_defs + log_p_away_atts + log_p_away_defs)
                   + np.sum(log_p_home_goals + log_p_away_goals))

    return log_density

def random_walk_metropolis(init_theta, init_eta, data, sigma, max_iter):
    """Random walk Metropolis algorithm for generating MCMC chains.

    Implements the random walk Metropolis algorithm using a Gaussian proposal distribution
    with isotropic covariance. The latter is an instance of a case whereby the proposal
    distribution is "symmetric", meaning that there is no "Hastings correction", and
    therefore this is the Metropolis algorithm, not the Metropolis-Hastings algorithm.

    Using initialisations for the parameters `init_theta` and the hyperparameters
    `init_eta`; observed `data` and the isotropic Gaussian standard deviation for the
    proposal distribution `sigma`, runs the Metropolis sampler for `max_iter` number
    of iterations.

    The random walk Metropolis algorithm implements the following:

    1. Initialise the parameters `init_theta` (and hyperparams `init_eta`) passed as args.
    2. Sample a candidate value of the parameters, theta_prime using the proposal distribution.

        `theta_prime` ~ Normal(`theta_prime` | `theta_prev`, (`sigma` ** 2) * I)

    3. Compute the log-acceptance probability.

        log(`alpha`) = min(0, log p(`theta_prime`, `init_eta`, `data`) - log p(`theta_prev`, `init_eta`, `data`))

    4. Sample a uniformly distributed random variable.

        `u` ~ Uniform(0, 1)

    5. Accept/reject decision rule.

        If log(`u`) <= log(`alpha`) then accept the sample, and set `theta[t+1, :]` <- `theta_prime`
        Otherwise reject teh sample and set `theta[t+1, :]` <- `theta_prev`

    Returns `theta`, an MCMC chain of the parameters as a shape-(`max_iter` + 1, 41)
    ndarray, where the no. of columns in axis-1 corresponds to the dimensionality/no.
    of individual parameters in `init_theta`. The first row of this array is the
    initialisation `init_theta`, and subsequent rows correspond to subsequent iterations
    in the MCMC chain.

    N.B. This implementation of the Metropolis algorithm does not infer the hyperparameters/
    latent variables `eta`.

    See "HW2 10-708 write-up.ipynb" for further info.

    Parameters
    ----------
    init_theta : ndarray
        A shape-(41,) ndarray - Initialisation of the model parameters.

    init_eta : ndarray
        A shape-(4,) ndarray - Initialisation of the model hyperparameters.

    data : ndarray
        A shape-(380, 4) int ndarray - Observed data.

    sigma : float
        Standard deviation parameter of the isotropic Gaussian proposal distribution.

    max_iter : int
        Number of iterations to run the MCMC chain.

    Returns
    -------
    out : tuple(ndarray, int)
        Tuple containing the following:

            1. theta : ndarray
                A shape-(max_iter + 1, 41) ndarray - MCMC chain of the model parameters.

            2. num_accepted_samples : int
                The number of samples that are acccepted by the Metropolis sampler.
    """

    dim = init_theta.shape[0]
    theta = np.zeros((max_iter + 1, dim))
    theta[0, :] = init_theta
    num_accepted_samples = 0

    print("Initial hyperparameter mu_att is: {}".format(init_eta[0]))
    print("Initial hyperparameter mu_def is: {}".format(init_eta[1]))
    print("Initial hyperparameter tau_att is: {}".format(init_eta[2]))
    print("Initial hyperparameter tau_def is: {}".format(init_eta[3]))

    for t in range(max_iter):

        print("=" * 30 + " t = {} ".format(t) + "=" * 30)

        # Sample the parameters from the isotropic Gaussian proposal distribution.
        theta_prime = multivariate_normal.rvs(mean=theta[t, :], cov=(sigma ** 2),
                                              size=1, random_state=None)

        # Enforce the corner-constraint on team-0 parameters.
        theta_prime[1] = 0
        theta_prime[21] = 0

        theta_prev = theta[t, :]

        # Compute the log-acceptance probability.
        log_density_theta_prime = compute_log_density(theta_prime,
                                                      init_eta,
                                                      data)

        log_density_theta_prev = compute_log_density(theta_prev,
                                                     init_eta,
                                                     data)

        log_alpha = np.minimum(0, log_density_theta_prime - log_density_theta_prev)

        print("log(alpha) is: {}".format(log_alpha))

        # Sample a uniformly distributed random variable.
        u = uniform.rvs(loc=0, scale=1, size=1, random_state=None)

        print("log(u) is: {}".format(np.log(u)))

        # Accept/reject decision rule.
        if np.log(u) <= log_alpha:
            print("As log(u) is less than or equal to log(alpha), we accept this sample from the proposal.")
            theta[t + 1, :] = theta_prime
            num_accepted_samples += 1
        else:
            print("As log(u) is greater than log(alpha), we reject this sample from the proposal")
            theta[t + 1, :] = theta[t, :]

    return theta, num_accepted_samples


# The following are helper functions to generate traceplots and an acceptance rate table.

def create_dict():
    """Construct team_idx -> team name dictionary

    Returns
    -------
    team_idx_name_dict : Dict[int, str]
        A dictionary of (team_ix -> team name) key-value pairs.
    """

    team_idx = [idx for idx in range(20)]
    names = ['Arsenal', 'Aston Villa', 'Cardiff City', 'Chelsea', 'Crystal Palace',
             'Everton', 'Fulham', 'Hull City', 'Liverpool', 'Manchester City',
             'Manchester United', 'Newcastle United', 'Norwich City', 'Southampton', 'Stoke City',
             'Sunderland', 'Swansea City', 'Tottenham Hotspurs', 'West Bromwich Albion', 'West Ham United']

    team_idx_name_dict = dict(zip(team_idx, names))

    return team_idx_name_dict


def create_new_sub_dir_paths():

    paths = dict()

    # Name of new sub-directories where scripts will be saved.
    sub_dir_names = ['logs', 'results', 'figures']

    # Get current working directory.
    cwd = os.getcwd()

    # Create new sub-directory paths.
    for name in sub_dir_names:
        path = os.path.join(cwd, name)
        paths[name] = path

    return paths

def create_new_sub_dirs(paths):

    for name in paths:
        sub_dir_path = paths[name]
        os.mkdir(sub_dir_path)

def generate_traceplots(png_filename, burn_in_samples, MCMC_samples,
                        max_burn_in_iter, num_samples, thinning_param, sigma):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(18, 10))
    total_iterations = max_burn_in_iter + (num_samples * thinning_param)
    post_samples = np.hstack((burn_in_samples, MCMC_samples))

    # Set time-series intervals.
    burn_in_t = np.arange(1 + max_burn_in_iter)
    MCMC_t = np.arange(num_samples * thinning_param)
    t = np.arange(1 + total_iterations)

    # Plot time-series.
    ax1.plot(t, post_samples)
    ax2.plot(burn_in_t, burn_in_samples)
    ax3.plot(MCMC_t, MCMC_samples)

    # Compute acceptance rate for the MCMC traceplot.
    acceptance_rate = np.unique(MCMC_samples).shape[0] / (num_samples * thinning_param)

    # Annotate plot.
    fig.suptitle(r"Traceplots for latent variable $home$."
                 + "\n"
                 + "MCMC parameters: " + r"$\sigma = ${}, ".format(sigma) + r"$t = $ {}.".format(thinning_param)
                 + "\n")

    ax1.set_title("Full traceplot: {} burn-in iterations, ".format(max_burn_in_iter)
                  + "{} MCMC sampler iterations.".format(num_samples * thinning_param))
    ax1.set_xlabel("Iterations.")

    ax2.set_title("Burn-in traceplot: {} burn-in iterations.".format(max_burn_in_iter))
    ax2.set_xlabel("Burn-in iterations.")

    ax3.set_title("MCMC traceplot: {} MCMC sampler iterations.".format(num_samples * thinning_param)
                  + "\n"
                  + "Estimated acceptance rate: {}".format(acceptance_rate))
    ax3.set_xlabel("MCMC sampler iterations.")

    for ax in [ax1, ax2, ax3]:
        ax.set_ylabel(r"$home$")

    # Leave whitespace between the plots.
    # Shade the burn-in phase red.
    fig.tight_layout(pad=1.08)
    ax1.axvspan(0, 5000, alpha=0.7, color='red')

    # Axes guidelines for all plots.
    for ax in [ax1, ax2, ax3]:
        ax.axhline(0, color='black', linewidth=0.6)
        ax.axvline(0, color='black', linewidth=0.6)

    # Save traceplot as a .png file.
    fig.savefig(png_filename)


def generate_table(png_filename, acceptance_rates, sigmas, thinning_params):

    fig, ax = plt.subplots()

    # Truncate the acceptance rate table to 5 decimal places.
    tr_acceptance_rates = np.round(acceptance_rates, decimals=5)

    # Create table.
    columns = ["$t = {}$".format(thinning_param) for thinning_param in thinning_params]
    rows = ["$\sigma = ${}".format(sigma) for sigma in sigmas]

    table = ax.table(cellText=tr_acceptance_rates,
                     rowLabels=rows, colLabels=columns,
                     loc='top', cellLoc='center')

    # Add table info.
    num_settings = len(sigmas) * len(thinning_params)
    plt.suptitle("Empirical acceptance rates on autocorrelated MCMC samples "
                 + "for {} settings of the MCMC parameters ".format(num_settings)
                 + r"$\sigma$ and t."
                 + "\n"
                 + "$\sigma$ : " + "Standard deviation of isotropic Gaussian proposal distribution"
                 + "; $t$ : " + "Thinning parameter.")

    # Hide background gridlines and add white-space.
    ax.axis('off')
    fig.tight_layout(pad=1.08)

    # Save the table as a .png file.
    # The bbox_inches **kwarg prevents the saved .png file from being inappropriately cropped.
    fig.savefig(png_filename, bbox_inches='tight')


if __name__ == "__main__":

    # Get current working directory and create 3 new sub-directories.
    # .txt log files will go into '/logs/
    # .npz results files will go into '/results'
    # .png figures will go into '/figures'
    paths = create_new_sub_dir_paths()
    create_new_sub_dirs(paths)

    # Keep a sys.stdout in a local variable to restore later.
    orig_stdout = sys.stdout

    # Create a global log-file recording brief details on the behaviour of the entire script.
    global_log_filename = "metropolis-log.txt"
    global_log_filepath = os.path.join(paths['logs'], global_log_filename)
    sys.stdout = open(global_log_filepath, 'w')

    # Set a random seed for the pseudo-random number generator.
    np.random.seed(21)

    # Initial run only.
    # Save the state of the PRNG to an .npy file in current working directory for reproducibility.
    # print("Saving the pseudo-random number generator state...")
    # PRNG_state = np.random.get_state()
    # PRNG_filename = "PRNG_state.npy"
    # np.save(PRNG_filename, PRNG_state)

    # For those attempting to reproduce the results.
    # Load the state of PRNG that was previously used to generate the results from 'PRNG_state.npy'
    print("Loading the saved pseudo-random number generator state...")
    PRNG_filename = "PRNG_state.npy"
    npy_file = np.load(PRNG_filename, allow_pickle=True)
    PRNG_state = tuple(npy_file)
    np.random.set_state(PRNG_state)

    # Load data.
    data = np.genfromtxt('premier_league_2013_2014.dat', dtype=int, delimiter=',')

    # Construct placeholders.
    num_teams = 20
    home_burn_in_init = np.zeros((1))
    atts_burn_in_init = np.zeros((num_teams))
    defs_burn_in_init = np.zeros((num_teams))

    # Initialise the hyper-hyperparameters, which parametrise the hyper-priors.
    tau_0 = 0.0001
    tau_1 = 0.0001
    alpha = 0.1
    beta = 0.1

    # Sample the hyperparameters from their hyper-priors.
    # These are fixed for the duration of the Metropolis-Hastings algorithm.
    mu_att = norm.rvs(loc=0, scale=(np.sqrt(tau_1)), size=1, random_state=None)
    mu_def = norm.rvs(loc=0, scale=(np.sqrt(tau_1)), size=1, random_state=None)
    tau_att = gamma.rvs(alpha, scale=(1 / beta), size=1, random_state=None)
    tau_def = gamma.rvs(alpha, scale=(1 / beta), size=1, random_state=None)

    # Place initialisations of the parameters and hyperparameters into vectors.
    eta_fixed = np.hstack((mu_att, mu_def, tau_att, tau_def))
    theta_burn_in_init = np.hstack((home_burn_in_init, atts_burn_in_init, defs_burn_in_init))

    # Set no. of iterations to burn-in the Metropolis algorithm.
    max_burn_in_iter = 5000

    # Initialise the grid of MCMC parameter settings for which to run Metropolis algorithm.
    sigmas, thinning_params, num_settings = parameter_grid()

    # Placeholder to store acceptance rates for each MCMC parameter setting.
    acceptance_rates = np.zeros((len(sigmas), len(thinning_params)))

    # Run the Metropolis algorithm for all settings of sigma, thinning_param.
    print("Running the Metropolis algorithm {} times ".format(num_settings)
          + "corresponding to each setting of the MCMC proposal variances"
          + "and thinning parameters.")

    start_time = datetime.now()

    for id, sigma in enumerate(sigmas):
        for id2, thinning_param in enumerate(thinning_params):

            # Redirect stdout to global log file.
            sys.stdout = open(global_log_filepath, 'a')
            print("Running the Metropolis algorithm with MCMC parameter setting  sigma = {}, t = {}...".format(sigma, thinning_param))
            sys.stdout.close()

            # Now redirect stdout to individual log files for each MCMC parameter setting.
            log_filename = "metropolis-log-sigma={}-t={}.txt".format(sigma, thinning_param)
            log_filepath = os.path.join(paths['logs'], log_filename)
            sys.stdout = open(log_filepath, 'w')

            print("=" * 100)
            print("=" * 100)
            print("Commencing the Metropolis algorithm for the following MCMC parameter setting:")
            print("sigma = {}".format(sigma))
            print("thinning parameter = {}".format(thinning_param))

            # Make a call to the ramdom seed for each run of the Metropolis algorithm.
            # Controlling for the inherent stochasticity of MCMC allows for fair comparison/
            # evaluation of Metropolis algorithm for each of the MCMC parameter settings.
            np.random.seed(21)

            # Burn-in the Metropolis algorithm.
            print("Burning-in the Metroplis algorithm for {} iterations...".format(max_burn_in_iter))
            theta_burn_in, _ = random_walk_metropolis(theta_burn_in_init, eta_fixed, data, sigma, max_burn_in_iter)
            print("=" * 100)
            print("=" * 100)
            print("Burn-in has concluded.")
            print("After burning-in for 5000 iterations, the parameters theta are: ")
            print(theta_burn_in)

            # Use the sample from the final iteration of the burn-in phase
            # as the initialisation for the MCMC/Metropolis sampler.
            theta_init = theta_burn_in[-1, :]

            # Set the number of samples to draw from the posterior after thinning.
            # Use with the thinning parameter to determine the number of autocorrelated
            # samples to generate from the MCMC/Metropolis sampler.
            num_samples = 5000
            max_iter = num_samples * thinning_param

            print("Generating {} autocorrelated samples from the Metropolis algorithm".format(max_iter))
            theta_MCMC_samples, num_accepted_samples = random_walk_metropolis(theta_init, eta_fixed, data,
                                                                              sigma, max_iter)

            acceptance_rate = num_accepted_samples / max_iter

            print("=" * 100)
            print("=" * 100)
            print("Metropolis algorithm has concluded for this setting of the MCMC parameters.")
            print("After {} iterations of the Metropolis algorithm, autocorrelated samples of the".format(max_iter)
                  + " parameters theta are: ")
            print(theta_MCMC_samples)
            print("\n")
            print("The acceptance rate is: {} accepted / {} total samples = {}".format(num_accepted_samples,
                                                                                        max_iter,
                                                                                        acceptance_rate))

            # As samples generated from the Metropolis algorithm are from a Markov chain, and
            # hence autocorrelated, we thin the samples, and collect every t-th sample
            # out of a total of max_iter samples, where t = thinning_param.
            thin_sample_idx = np.array([t for t in range(thinning_param, max_iter + 1, thinning_param)]) - 1

            # Exclude initialisation from the posterior sampling.
            theta_posterior_samples = theta_MCMC_samples[1:, :][thin_sample_idx]
            print("After thinning, and assuming that the Markov chain has converged to a stationary distribution "
                  "that is the target/posterior distribution, the posterior samples of the parameters "
                  "theta are: ")
            print(theta_posterior_samples)

            # Store the acceptance rate for this MCMC parameter setting into the placeholder.
            acceptance_rates[id, id2] = acceptance_rate

            # Save initialisations, burn-in history, MCMC sampling history, acceptance rate
            # and posterior samples to a file for further analysis.
            print("Saving results to an .npz file...")
            npz_filename = ("metropolis-results-" + "sigma={}-".format(sigma)
                            + "t={}.npz".format(thinning_param))

            names = ["eta_fixed", "theta_burn_in_init", "theta_burn_in",
                     "theta_MCMC_samples", "theta_posterior_samples", "acceptance_rate"]

            arrays = [eta_fixed, theta_burn_in_init, theta_burn_in,
                      theta_MCMC_samples, theta_posterior_samples, acceptance_rate]

            kwargs = dict()
            for name, array in zip(names, arrays):
                kwargs[name] = array

            npz_filepath = os.path.join(paths['results'], npz_filename)

            np.savez(npz_filepath, **kwargs)

            # Generate and save traceplots for the home advantage parameter, as per assignment requirements.
            # The other visualisations require analysis first.
            print("Generating trace plots of latent variable `home` for the current MCMC parameter setting...")
            png_filename = ('traceplots-home-sigma={}-t={}.png'.format(sigma, thinning_param))
            png_filepath = os.path.join(paths['figures'], png_filename)
            generate_traceplots(png_filepath, theta_burn_in[:, 0], theta_MCMC_samples[1:, 0],
                                max_burn_in_iter, num_samples, thinning_param, sigma)

            # Complete writing to the individual log file.
            sys.stdout.close()

    sys.stdout = open(global_log_filepath, 'a')
    print("Metropolis algorithm has now concluded for each of the {} MCMC parameter settings".format(num_settings))

    # Generate and save the acceptance rate table for all MCMC parameter settings,
    # as per assignment requirements.
    print("Generating table of acceptance rates computed from each MCMC parameter setting...")
    table_filename = "acceptance-rate-table.png"
    table_filepath = os.path.join(paths['figures'], table_filename)
    generate_table(table_filepath, acceptance_rates, sigmas, thinning_params)

    duration = datetime.now() - start_time
    print("Total time elapsed: {}".format(duration))






