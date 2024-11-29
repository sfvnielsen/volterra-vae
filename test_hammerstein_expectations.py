"""
    Script that tests the pen-and-paper derivations of the expectations from the Hammerstein VAE
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numba import njit, prange, set_num_threads



# Numba functions for computing the empirical values of the cross-term (h x h x^2)
@njit(parallel=True)
def sample_bernoulli_cross_term(result_array, p, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = random_generator.uniform(size=len(p)) <= p
        x = x.astype(np.float64)
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += h @ x[j:nlags+j][::-1] * (h @ (x[j:nlags+j][::-1])**2)
        result_array[i] = ysum

@njit(parallel=True)
def sample_normal_cross_term(result_array, mu, var, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = np.sqrt(var) * random_generator.standard_normal(size=len(mu)) + mu
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += h @ x[j:nlags+j][::-1] * (h @ (x[j:nlags+j][::-1])**2)
        result_array[i] = ysum


# Numba functions for computing the empirical values of the square-square term ( (h x^2)^2)
@njit(parallel=True)
def sample_bernoulli_square2_term(result_array, p, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = random_generator.uniform(size=len(p)) <= p
        x = x.astype(np.float64)
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (h @ (x[j:nlags+j][::-1])**2)**2
        result_array[i] = ysum

@njit(parallel=True)
def sample_normal_square2_term(result_array, mu, var, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = np.sqrt(var) * random_generator.standard_normal(size=len(mu)) + mu
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (h @ (x[j:nlags+j][::-1])**2)**2
        result_array[i] = ysum


FIGSIZE = (12.5, 7.5)
DPI = 150
FIGURE_DIR = 'results/figures'
FIGPREFIX = 'hammer_expect'


def plot_expecation_distribution(samples, standard_error, ax, rel_text_placement, n_histogram_bins=1000):
    emp_mean = np.mean(samples)
    ax.hist(samples, bins=n_histogram_bins)
    ax.axvline(x=derived, color='r')
    ax.axvline(x=emp_mean, color='b', linestyle='--')

    ax.text(rel_text_placement[0], rel_text_placement[1], f'Emp. mean: {emp_mean:.5f}\n Standard error {standard_error:.5f}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, color='b')
    ax.text(rel_text_placement[0], rel_text_placement[1] - 0.1, f'Derived: {derived:.5f}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, color='r')

    if (derived < (emp_mean + 1.96 * standard_error)) and (derived > (emp_mean - 1.96 * standard_error)):
        ax.text(rel_text_placement[0], rel_text_placement[1] - 0.3, "Theory is within 95pct conf", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, color='k')


if __name__ == "__main__":
    # Simulation parameters
    SEQUENCE_LENGTH = 25
    KERNEL_MEMORY = 7
    NDRAWS = int(1e6)
    seeds = np.arange(0, 10)
    set_num_threads(1)  # fix to 1 for proper randomization (setting this higher results in weird random object behaviour)

    # Choose which distribution the input signal should follow
    x_distribution = 'bernoulli'  # supported are 'normal' or 'bernoulli'

    # Loop over seeds - plot the distribution of the samples together with theoretical expectation
    n_trials = len(seeds)
    n_plots_per_row = 4
    fig_ct, ax_ct = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    ax_ct = ax_ct.flatten()
    fig_sqsq, ax_sqsq = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    ax_sqsq = ax_sqsq.flatten()
    n_histogram_bins = 1000
    rel_text_placement = (0.5, 0.75)

    figpref = f'{FIGPREFIX}_{x_distribution}_seqlength_{SEQUENCE_LENGTH}'

    for s, seed in enumerate(seeds):
        print(f"Seed is {seed}")
        print(f"Drawing {NDRAWS:.2e} samples of length {SEQUENCE_LENGTH} from the {x_distribution} distribution.")

        # Initialize random object - for generating parameters
        random_obj = np.random.default_rng(seed)

        # Generate random FIR filter
        h = random_obj.uniform(size=KERNEL_MEMORY)

        # Generate distribution parameters
        if x_distribution == 'bernoulli':
            p = random_obj.uniform(size=SEQUENCE_LENGTH)
        elif x_distribution == 'normal':
            mu = random_obj.standard_normal(size=SEQUENCE_LENGTH)
            precision = random_obj.standard_gamma(shape=1.0, size=SEQUENCE_LENGTH)
            variance = 1 / precision
        else:
            raise Exception(f"Unknown distribution for x : '{x_distribution}'")

        # Define raw moments - used in formula(s) below
        if x_distribution == 'bernoulli':
            ex = p  # expectation of x (bernoulli) - all raw moments are equal
            ex2 = p
            ex3 = p
            ex4 = p
        elif x_distribution == 'normal':
            ex = mu
            ex2 = mu ** 2 + variance
            ex3 = mu ** 3 + 3 * mu * variance
            ex4 = mu ** 4 + 6 * mu ** 2 * variance + 3 * variance ** 2

        """
                Cross term - E[h_i x_i h_j x_j^2]
        """

        print('Computing cross term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,))

        if x_distribution == 'bernoulli':
            sample_bernoulli_cross_term(empirical, p, h, random_obj)
        elif x_distribution == 'normal':
            sample_normal_cross_term(empirical, mu, variance, h, random_obj)
        else:
            raise ValueError

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[h_i x_i h_j x_j^2]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        derived = np.sum(np.convolve(ex2, h, 'valid') * np.convolve(ex, h, 'valid') + np.convolve(ex3 - ex2 * ex, h**2, 'valid'))
        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")

        plot_expecation_distribution(empirical, standard_error, ax_ct[s], rel_text_placement)


        """
                Squared-square term - E[(h_i x_i^2)^2]
        """

        print('Computing cross term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,))

        if x_distribution == 'bernoulli':
            sample_bernoulli_square2_term(empirical, p, h, random_obj)
        elif x_distribution == 'normal':
            sample_normal_square2_term(empirical, mu, variance, h, random_obj)
        else:
            raise ValueError

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[(h_i x_i^2)^2]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        derived = np.sum(np.convolve(ex2, h, 'valid')**2 + np.convolve(ex4 - ex2**2, h**2, 'valid'))
        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")

        plot_expecation_distribution(empirical, standard_error, ax_sqsq[s], rel_text_placement)

        

    print("Done.")

    # Set some figure properties.
    fig_ct.suptitle('Cross terms E[x_i x_j x_k h_i H_jk]')

    fig_ct.savefig(os.path.join(FIGURE_DIR, f"{figpref}_cross_terms.png"), dpi=DPI)

    plt.show()
