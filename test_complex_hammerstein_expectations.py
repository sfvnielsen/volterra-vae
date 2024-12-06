"""
    Script that tests the pen-and-paper derivations of the expectations from the Hammerstein VAE
    Complex valued input
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numba import njit, prange, set_num_threads


# Numba function for computing the empirical values of the cross-term Re( (h x)^* h x^2)
@njit(parallel=True)
def sample_bernoulli_cross_term(result_array, p_re, p_im, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        xre = random_generator.uniform(size=len(p_re)) <= p_re
        xim = 1j * (random_generator.uniform(size=len(p_im)) <= p_im)
        x = (xre + xim).astype(np.complex128)
        ysum = 0.0
        for j in range(len(p_re) - nlags + 1):
            ysum += np.real(np.conjugate(h @ x[j:nlags+j][::-1]) * (h @ (x[j:nlags+j][::-1])**2))
        result_array[i] = ysum

# Numba function for computing the empirical values of the first squared term E[(h x)^* h x)]
@njit(parallel=True)
def sample_bernoulli_squared_first_term(result_array, p_re, p_im, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        xre = random_generator.uniform(size=len(p_re)) <= p_re
        xim = 1j * (random_generator.uniform(size=len(p_im)) <= p_im)
        x = (xre + xim).astype(np.complex128)
        ysum = 0.0
        for j in range(len(p_re) - nlags + 1):
            ysum += np.real(np.conjugate(h @ x[j:nlags+j][::-1]) * (h @ (x[j:nlags+j][::-1])))
        result_array[i] = ysum

# Numba function for computing the empirical values of the second squared term E[(h x^2)^* h x^2)]
@njit(parallel=True)
def sample_bernoulli_squared_second_term(result_array, p_re, p_im, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        xre = random_generator.uniform(size=len(p_re)) <= p_re
        xim = 1j * (random_generator.uniform(size=len(p_im)) <= p_im)
        x = (xre + xim).astype(np.complex128)
        ysum = 0.0
        for j in range(len(p_re) - nlags + 1):
            ysum += np.real(np.conjugate(h @ x[j:nlags+j][::-1]**2) * (h @ (x[j:nlags+j][::-1]**2)))
        result_array[i] = ysum



FIGSIZE = (12.5, 7.5)
DPI = 150
FIGURE_DIR = 'results/figures'
FIGPREFIX = 'hammer_complex_expect'


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

    # Loop over seeds - plot the distribution of the samples together with theoretical expectation
    n_trials = len(seeds)
    n_plots_per_row = 4
    fig_ct, ax_ct = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    ax_ct = ax_ct.flatten()
    fig_fisq, ax_fisq = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    ax_fisq = ax_fisq.flatten()
    fig_sesq, ax_sesq = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    ax_sesq = ax_sesq.flatten()
    n_histogram_bins = 1000
    rel_text_placement = (0.5, 0.75)

    figpref = f'{FIGPREFIX}_seqlength_{SEQUENCE_LENGTH}'

    for s, seed in enumerate(seeds):
        print(f"Seed is {seed}")
        print(f"Drawing {NDRAWS:.2e} samples of length {SEQUENCE_LENGTH}.")

        # Initialize random object - for generating parameters
        random_obj = np.random.default_rng(seed)

        # Generate random FIR filter
        h = random_obj.uniform(size=KERNEL_MEMORY) + 1j * random_obj.uniform(size=KERNEL_MEMORY)

        # Generate distribution parameters
        pre = random_obj.uniform(size=SEQUENCE_LENGTH)
        pim = random_obj.uniform(size=SEQUENCE_LENGTH)
        
        # Define raw moments - used in formula(s) below
        ex_i, ex_q = pre, pim # expectation of x (bernoulli) - all raw moments are equal
        ex2_i, ex2_q = pre, pim
        ex3_i, ex3_q = pre, pim 
        ex4_i, ex4_q = pre, pim 

        """
                Cross term - E[Re( (h_i x_i)^* h_j x_j^2)]
        """

        print('Computing cross term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,), dtype=np.float64)

        # Call numba routine
        sample_bernoulli_cross_term(empirical, pre, pim, h, random_obj)

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[Re( (h_i x_i)^* h_j x_j^2)]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        d1i = np.convolve(np.real(h), ex_i, 'valid') - np.convolve(np.imag(h), ex_q, 'valid')
        d1q = np.convolve(np.real(h), ex_q, 'valid') + np.convolve(np.imag(h), ex_i, 'valid')
        d2i = np.convolve(np.real(h), ex2_i - ex2_q, 'valid') - np.convolve(np.imag(h), 2 * ex_i * ex_q, 'valid')
        d2q = np.convolve(np.real(h), 2 * ex_i * ex_q, 'valid') + np.convolve(np.imag(h), ex2_i - ex2_q, 'valid')
        derived = np.sum(d1i * d2i + d1q * d2q + np.convolve(np.abs(h)**2, ex3_i - ex_i * ex2_i + 2*(ex_i * ex2_q - ex_i * ex_q**2), 'valid'))
        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")

        plot_expecation_distribution(empirical, standard_error, ax_ct[s], rel_text_placement)

        """
                First squared term - E[(h_i x_i)^* (h_j x_j))] - identical to Lauinger et al 2022 (E term)
        """

        print('Computing first squared term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,), dtype=np.float64)

        # Call numba routine
        sample_bernoulli_squared_first_term(empirical, pre, pim, h, random_obj)

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[(h x)* (h x)]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        derived = np.sum(np.square(d1i) + np.square(d1q) + np.convolve(np.abs(h)**2, ex2_i + ex2_q - ex_i**2 - ex_q**2, 'valid'))
        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")

        plot_expecation_distribution(empirical, standard_error, ax_fisq[s], rel_text_placement)

        """
                Second squared term - E[(h_i x_i^2)^* (h_j x_j^2))]
        """

        print('Computing second squared term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,), dtype=np.float64)

        # Call numba routine
        sample_bernoulli_squared_second_term(empirical, pre, pim, h, random_obj)

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[(h x^2)* (h x)^2]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        derived = np.sum(np.square(d2i) + np.square(d2q) + np.convolve(np.abs(h)**2, ex4_i + ex4_q - ex2_i**2 - ex2_q**2 + 4*ex2_i * ex2_q - 4*ex_i**2*ex_q**2, 'valid'))
        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")

        plot_expecation_distribution(empirical, standard_error, ax_sesq[s], rel_text_placement)
        

    print("Done.")

    # Set some figure properties.
    fig_ct.suptitle('Cross terms E[Re(hx* hx2)]')
    fig_fisq.suptitle('First squared term E[(hx* hx)]')
    fig_sesq.suptitle('Second squared term E[(hx^2)* (hx^2)]')

    plt.show()
