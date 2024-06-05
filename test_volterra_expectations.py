"""
    Script that tests the pen-and-paper derivations of the expectations from the Volterra VAE
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from numba import njit, prange, set_num_threads


# Numba functions for computing the empirical values of the 1st order kernel applied to the signal (h x) squared
@njit(parallel=True)
def sample_bernoulli_first_term(result_array, p, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
#    for i in range(len(result_array)):
        x = random_generator.uniform(size=len(p)) <= p
        x = x.astype(np.float64)
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (h @ x[j:nlags+j][::-1]) ** 2
        result_array[i] = ysum

@njit(parallel=True)
def sample_normal_first_term(result_array, mu, var, h, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = np.sqrt(var) * random_generator.standard_normal(size=len(mu)) + mu
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (h @ x[j:nlags+j][::-1]) ** 2
        result_array[i] = ysum


# Numba functions for computing the empirical values of the cross-term (h x xT H x)
@njit(parallel=True)
def sample_bernoulli_cross_term(result_array, p, h, H, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = random_generator.uniform(size=len(p)) <= p
        x = x.astype(np.float64)
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (h @ x[j:nlags+j][::-1]) * x[j:nlags+j][::-1].T @ (H @ x[j:nlags+j][::-1])
        result_array[i] = ysum

@njit(parallel=True)
def sample_normal_cross_term(result_array, mu, var, h, H, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = np.sqrt(var) * random_generator.standard_normal(size=len(mu)) + mu
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (h @ x[j:nlags+j][::-1]) * x[j:nlags+j][::-1].T @ (H @ x[j:nlags+j][::-1])
        result_array[i] = ysum

# Numba functions for computing the empirical values of the squared-term (xT H x)^2
@njit(parallel=True)
def sample_bernoulli_squared_term(result_array, p, H, random_generator):
    nlags = len(h)
    for i in prange(len(result_array)):
        x = random_generator.uniform(size=len(p)) <= p
        x = x.astype(np.float64)
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (x[j:nlags+j][::-1].T @ (H @ x[j:nlags+j][::-1])) ** 2
        result_array[i] = ysum

@njit(parallel=True)
def sample_normal_squared_term(result_array, mu, var, H, random_generator):
    nlags = len(h)
    for i in prange(len(result_array) - nlags):
        x = np.sqrt(var) * random_generator.standard_normal(size=len(mu)) + mu
        ysum = 0.0
        for j in range(len(p) - nlags + 1):
            ysum += (x[j:nlags+j][::-1].T @ (H @ x[j:nlags+j][::-1])) ** 2
        result_array[i] = ysum


FIGSIZE = (12.5, 7.5)
DPI = 150
FIGURE_DIR = 'results/figures'
FIGPREFIX = 'vol_expect'


if __name__ == "__main__":
    # Simulation parameters
    SEQUENCE_LENGTH = 25
    KERNEL_MEMORY = 5
    NDRAWS = int(1e6)
    #seeds = np.arange(0, 10)
    seeds = np.arange(10, 20)
    set_num_threads(1)  # fix to 1 for proper randomization (setting this higher results in weird random object behaviour)

    # Choose which distribution the input signal should follow
    x_distribution = 'bernoulli'  # supported are 'normal' or 'bernoulli'

    # Choose if we run with symmetric second order kernel
    symmetric_second_order_kernel = True

    # Loop over seeds - plot the distribution of the samples together with theoretical expectation
    n_trials = len(seeds)
    n_plots_per_row = 4
    fig_ct, ax_ct = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    fig_sqt, ax_sqt = plt.subplots(ncols=n_plots_per_row, nrows=n_trials // n_plots_per_row + 1, figsize=(15, 9))
    ax_ct = ax_ct.flatten()
    ax_sqt = ax_sqt.flatten()
    n_histogram_bins = 1000
    rel_text_placement = (0.5, 0.75)

    figpref = f'vol_expect_{x_distribution}_seqlength_{SEQUENCE_LENGTH}'
    if symmetric_second_order_kernel:
        figpref += '_symH'

    for s, seed in enumerate(seeds):
        print(f"Seed is {seed}")
        print(f"Drawing {NDRAWS:.2e} samples of length {SEQUENCE_LENGTH} from the {x_distribution} distribution.")

        # Initialize random object - for generating parameters
        random_obj = np.random.default_rng(seed)

        # Generate random Volterra kernels (h and H)
        h = random_obj.uniform(size=KERNEL_MEMORY)
        H = random_obj.uniform(size=(KERNEL_MEMORY, KERNEL_MEMORY))

        if symmetric_second_order_kernel:
            H = np.triu(H) + np.triu(H).T
            print("Second order term is symmtric")

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
                1st order term - E[x_i x_j h_i h_j]
        """
        print('Computing 1st order term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,))

        if x_distribution == 'bernoulli':
            sample_bernoulli_first_term(empirical, p, h, random_obj)
        elif x_distribution == 'normal':
            sample_normal_first_term(empirical, mu, variance, h, random_obj)
        else:
            raise ValueError

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[x_i x_j h_i h_j]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        derived = np.sum(np.convolve(ex, h, mode='valid')**2 + np.convolve(ex2 - ex**2, h**2, mode='valid'))

        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")


        """
                Cross term - E[x_i x_j x_k h_i H_jk]
        """

        print('Computing cross term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,))

        if x_distribution == 'bernoulli':
            sample_bernoulli_cross_term(empirical, p, h, H, random_obj)
        elif x_distribution == 'normal':
            sample_normal_cross_term(empirical, mu, variance, h, H, random_obj)
        else:
            raise ValueError

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[x_i x_j x_k h_i H_jk]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        ex_lag = toeplitz(ex, np.zeros(KERNEL_MEMORY, ))[(KERNEL_MEMORY-1):]
        ex2_lag = toeplitz(ex2, np.zeros(KERNEL_MEMORY, ))[(KERNEL_MEMORY-1):]
        ex2ex = np.einsum('ni,nj->ij', ex2_lag, ex_lag)
        exsqex = np.einsum('ni,nj->ij', ex_lag**2, ex_lag)
        if symmetric_second_order_kernel:
            x2x_sum = np.sum((ex2ex - exsqex) * (2 * np.einsum('i,ji->ij', h, H) + np.einsum('j,ii->ij', h, H)))
        else:
            x2x_sum = np.sum((ex2ex - exsqex) * (np.einsum('i,ji->ij', h, H) + np.einsum('i,ij->ij', h, H) + np.einsum('j,ii->ij', h, H)))
        
        x3_sum = np.sum(np.convolve(ex3 - 3 * ex2 * ex + 2*ex**3, h * np.diag(H), mode='valid'))

        derived =  np.sum(np.einsum('ni,nj,nk->ijk', ex_lag, ex_lag, ex_lag) * np.einsum('i,jk->ijk', h, H)) + x2x_sum + x3_sum

        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end- start} s")

        ax_ct[s].hist(empirical, bins=n_histogram_bins)
        ax_ct[s].axvline(x=derived, color='r')
        ax_ct[s].axvline(x=np.mean(empirical), color='b', linestyle='--')

        ax_ct[s].text(rel_text_placement[0], rel_text_placement[1], f'Emp. mean: {emp_mean:.5f}\n Standard error {standard_error:.5f}', horizontalalignment='center',
                      verticalalignment='center', transform=ax_ct[s].transAxes, color='b')
        ax_ct[s].text(rel_text_placement[0], rel_text_placement[1] - 0.1, f'Derived: {derived:.5f}', horizontalalignment='center',
                      verticalalignment='center', transform=ax_ct[s].transAxes, color='r')

        if (derived < (emp_mean + 1.96 * standard_error)) and (derived > (emp_mean - 1.96 * standard_error)):
            ax_ct[s].text(rel_text_placement[0], rel_text_placement[1] - 0.3, "Theory is within 95pct conf", horizontalalignment='center',
                      verticalalignment='center', transform=ax_ct[s].transAxes, color='k')

        """

               Squared term - E[x_i x_j x_k x_l H_ij H_kl]

        """
        print('Computing squared term expectation...')
        start = time.time()
        empirical = np.zeros((NDRAWS,))

        if x_distribution == 'bernoulli':
            sample_bernoulli_squared_term(empirical, p, H, random_obj)
        elif x_distribution == 'normal':
            sample_normal_squared_term(empirical, mu, variance, H, random_obj)
        else:
            raise ValueError

        emp_mean = np.mean(empirical)
        print(f"Empirical expectation of E[x_i x_j x_k x_l H_ij H_kl]: {emp_mean}")
        standard_error = np.std(empirical) / np.sqrt(NDRAWS)
        print(f"Associated standard error: {standard_error}")

        # Calculate expectation from derived formula
        ex_lag = toeplitz(ex, np.zeros(KERNEL_MEMORY, ))[(KERNEL_MEMORY-1):]
        ex2_lag = toeplitz(ex2, np.zeros(KERNEL_MEMORY, ))[(KERNEL_MEMORY-1):]
        ex3_lag = toeplitz(ex3, np.zeros(KERNEL_MEMORY, ))[(KERNEL_MEMORY-1):]

        xxxx_sum = np.sum(np.einsum('ni,nj,nk,nl->ijkl', ex_lag, ex_lag, ex_lag, ex_lag) * np.einsum('ij,kl->ijkl', H, H))

        ex2exex = np.einsum('ni,nj,nk->ijk', ex2_lag, ex_lag, ex_lag)
        exsqexex = np.einsum('ni,nj,nk->ijk', ex_lag**2, ex_lag, ex_lag)
        if symmetric_second_order_kernel:
            x2xx_sum = np.sum(np.multiply(ex2exex - exsqexex,
                                      2 * np.einsum('ii,jk->ijk', H, H) + 4 * np.einsum('ij,ik->ijk', H, H)))
        else:
            x2xx_sum = np.sum(np.multiply(ex2exex - exsqexex,
                                        np.einsum('ii,jk->ijk', H, H) + np.einsum('ij,ik->ijk', H, H) + np.einsum('ij,ki->ijk', H, H) + np.einsum('ji,ik->ijk', H, H) + np.einsum('ji,ki->ijk', H, H) + np.einsum('jk,ii->ijk', H, H)))
            

        ex2ex2 = np.einsum('ni,nj->ij', ex2_lag, ex2_lag)
        ex2exsq = np.einsum('ni,nj->ij', ex2_lag, ex_lag**2)
        exsqex2 = np.einsum('ni,nj->ij', ex_lag**2, ex2_lag)
        exsqexsq = np.einsum('ni,nj->ij', ex_lag**2, ex_lag**2)
        if symmetric_second_order_kernel:
            x2x2_sum = np.sum(np.multiply(ex2ex2 - ex2exsq - exsqex2 + exsqexsq,
                              2 * np.einsum('ij,ij->ij', H, H) + np.einsum('ii,jj->ij', H, H)))
        else:
            x2x2_sum = np.sum(np.multiply(ex2ex2 - ex2exsq - exsqex2 + exsqexsq,
                            np.einsum('ij,ij->ij', H, H) + np.einsum('ii,jj->ij', H, H) + np.einsum('ij,ji->ij', H, H)))
            
        ex3ex = np.einsum('ni,nj->ij', ex3_lag, ex_lag)
        ex2twoex = np.einsum('ni,ni,nj->ij', ex2_lag, ex_lag, ex_lag)
        excubex = np.einsum('ni,nj->ij', ex_lag**3, ex_lag)
        if symmetric_second_order_kernel:
            x3x_sum = np.sum(np.multiply(ex3ex - 3 * ex2twoex + 2 * excubex,
                                     4 * np.einsum('ii,ij->ij', H, H)))
        else:
            x3x_sum = np.sum(np.multiply(ex3ex - 3 * ex2twoex + 2 * excubex,
                                        np.einsum('ii,ij->ij', H, H) + np.einsum('ii,ji->ij', H, H) + np.einsum('ij,ii->ij', H, H) + np.einsum('ji,ii->ij', H, H)))

        x4_sum = np.sum(np.convolve(ex4 + 12 * ex2 * ex ** 2 - 3 * ex2 ** 2 - 4 * ex3 * ex - 6 * ex ** 4, np.diag(H)**2, mode='valid'))

        derived = xxxx_sum + x2xx_sum + x2x2_sum + x3x_sum + x4_sum


        print(f"Theoretial expectation: {derived}")

        end = time.time()
        print(f"Time elapsed: {end - start} s")

        ax_sqt[s].hist(empirical, bins=n_histogram_bins)
        ax_sqt[s].axvline(x=derived, color='r')
        ax_sqt[s].axvline(x=np.mean(empirical), color='b', linestyle='--')

        ax_sqt[s].text(rel_text_placement[0], rel_text_placement[1], f'Emp. mean: {emp_mean:.5f}\n Standard error {standard_error:.5f}', horizontalalignment='center',
                       verticalalignment='center', transform=ax_sqt[s].transAxes, color='b')
        ax_sqt[s].text(rel_text_placement[0], rel_text_placement[1] - 0.1, f'Derived: {derived:.5f}', horizontalalignment='center',
                       verticalalignment='center', transform=ax_sqt[s].transAxes, color='r')

        if (derived < (emp_mean + 1.96 * standard_error)) and (derived > (emp_mean - 1.96 * standard_error)):
            ax_sqt[s].text(rel_text_placement[0], rel_text_placement[1] - 0.3, "Theory is within 95pct conf", horizontalalignment='center',
                        verticalalignment='center', transform=ax_sqt[s].transAxes, color='k')

    print("Done.")

    # Set some figure properties.
    fig_ct.suptitle('Cross terms E[x_i x_j x_k h_i H_jk]')
    fig_sqt.suptitle('Squared terms E[x_i x_j x_k x_l H_ij H_kl]')

    fig_ct.savefig(os.path.join(FIGURE_DIR, f"{figpref}_cross_terms.png"), dpi=DPI)
    fig_sqt.savefig(os.path.join(FIGURE_DIR, f"{figpref}_squared_terms.png"), dpi=DPI)

    plt.show()
