import subprocess
import json
import numpy as np
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.stats import norm

API_KEY_FILE = '.api_keys.json'

def get_git_info():
    gitsha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode("utf-8")
    gitbranch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode("utf-8")
    return gitbranch, gitsha


def get_api_key(key_from_args=None):
    if key_from_args:
        return key_from_args

    # If api key was not supplied in command-line arguments
    # try getting the key from the api key file
    with open(API_KEY_FILE, 'r') as handle:
        api_keys = json.load(handle)
        wandb_api_key = api_keys["wandb"]

    return wandb_api_key


def symbol_sync(rx, tx, sps):
    """ Synchronizes tx symbols to the received signal, assumed to be oversample at sps
        Assumes that rx has been through a sampling recovery mechanism first

        Heavily inspired by the DSP library in https://github.com/edsonportosilva/OptiCommPy
    """
    rx_syms = rx[0::sps]
    delay = find_delay(tx, rx_syms)
    return np.roll(tx, -int(delay))


def find_delay(x, y):
    """ Find delay that maximizes correlation between real-parts

        Heavily inspired by the DSP library in https://github.com/edsonportosilva/OptiCommPy
    """
    return np.argmax(correlate(np.real(x), np.real(y))) - x.shape[0] + 1


def find_max_variance_sample(y, sps):
    """ Find sampling recovery compensation shift using the maximum variance method

        Heavily inspired by the DSP library in https://github.com/edsonportosilva/OptiCommPy
    """
    nsyms = len(y) // sps  # truncate to an integer num symbols
    yr = np.reshape(y[0:nsyms*sps], (-1, sps))
    var = np.var(yr, axis=0)
    max_variance_sample = np.argmax(var)
    return max_variance_sample


def decode_from_probs(q, constellation):
    """
        q is a matrix of probabilities (M x N), where N is the sequence 
        length and M is the constellation size
        
        return most likely symbol sequence (argmax pr. timepoint) 
    """
    assert q.shape[0] == len(constellation)
    best_indices = np.argmax(q, axis=0)
    return constellation[best_indices]


def calc_ser_from_probs(q, a, constellation, discard=10):
    ahat = decode_from_probs(q, constellation)
    opt_delay = find_delay(ahat, a)
    errors = ahat[discard:-discard] != np.roll(a, opt_delay)[discard:-discard]
    print(f"Total number of erros {np.sum(errors)} (optimal delay: {opt_delay})")
    return np.mean(errors), opt_delay


def calc_ser_pam(y_eq, a, discard=10):
    assert len(y_eq) == len(a)
    opt_delay = find_delay(y_eq, a)
    const = np.unique(a)
    ahat = decision_logic(y_eq, const, const)
    errors = ahat[discard:-discard] != np.roll(a, opt_delay)[discard:-discard]
    print(f"Total number of erros {np.sum(errors)} (optimal delay: {opt_delay})")
    return np.mean(errors), opt_delay


def calc_ser(ahat, a):
    assert (len(ahat) == len(a))

    def compare(x, y):
        return x != y

    i_errors = compare(np.real(ahat), np.real(a))
    q_errors = compare(np.imag(ahat), np.imag(a))
    total_errors = np.sum(i_errors) + np.sum(q_errors) - np.sum(np.logical_and(i_errors, q_errors))
    if total_errors < 10:
        print(f"WARNING! Total errors in SER calculation was {total_errors}")
    return total_errors / len(a)


def calc_robust_ser(y_eq, a, angles=[0.0, np.pi / 2, np.pi, np.pi * 3 / 2], delay_order=30, sublength=1000, discard=10):
    # Calculates symbol error rate, robust towards uknown delay and rotational ambiguities
    assert (y_eq.shape == a.shape)
    opt_delay = find_delay(y_eq, a)
    const = np.unique(a)
    sers = np.zeros_like(angles)
    for i, ang in enumerate(angles):
        ahat_rot = decision_logic(y_eq * np.exp(1j * ang), const, const)
        sers[i] = calc_ser(ahat_rot[discard:-discard], np.roll(a, opt_delay)[discard:-discard])
    return np.min(sers), opt_delay


def calc_brute_force_ser(y_eq, a, angles=[0.0, np.pi / 2, np.pi, np.pi * 3 / 2], delay_order=30, discard=10):
    # Calculates symbol error rate, robust towards uknown delay and rotational ambiguities by trying all the combinations
    assert (y_eq.shape == a.shape)
    const = np.unique(a)
    delays = np.arange(-delay_order, delay_order + 1)
    sers = np.empty((len(angles), len(delays)))
    for i, ang in enumerate(angles):
        for j, delay in enumerate(delays):
            ahat_rot = decision_logic(y_eq * np.exp(1j * ang), const, const)
            sers[i, j] = calc_ser(ahat_rot[discard:-discard], np.roll(a, delay)[discard:-discard])
    min_ser_idx = np.unravel_index(np.argmin(sers), sers.shape)
    opt_delay = delays[min_ser_idx[1]]
    opt_rot = angles[min_ser_idx[0]]
    return np.min(sers), opt_delay, opt_rot


def calc_theory_ser_qam(constellation_order, EsN0_db):
    # Theoretical SER in a square even QAM constellation (4, 16, 64, ...)
    # https://dsp.stackexchange.com/questions/15996/how-is-the-symbol-error-rate-for-m-qam-4qam-16qam-and-32qam-derived
    assert (int(np.log2(constellation_order)) % 2 == 0)
    ser_theory = 0.0
    snr_lin = 10 ** (EsN0_db / 10.0)
    qx = norm.sf(np.sqrt(snr_lin))  # complementary cumulative pdf of standard gaussain
    corner_points, edge_points, interior_points = 4, 0, 0

    ser_theory += corner_points * (2 * qx - qx**2)

    # If QAM order is higher than four, count edge and interior points also
    if constellation_order != 4:
        n = int(np.log2(constellation_order)) // 2
        edge_points = 4 * (2 ** n - 2)
        ser_theory += edge_points * (3 * qx - 2 * qx**2)

        interior_points = (2**n - 2) ** 2
        ser_theory += interior_points * (4 * qx - 4 * qx**2)

    return ser_theory / constellation_order


def calc_theory_ser_pam(constellation_order, EsN0_db):
    # Theorertical SER in a PAM constellation
    # Taken from Holger Krener Iversen's thesis
    snr_lin = 10 ** (EsN0_db / 10.0)
    qx = norm.sf(np.sqrt(snr_lin))  # complementary cumulative pdf of standard gaussain
    return 2 * (constellation_order - 1) / constellation_order * qx


def decision_logic(xhat, syms, symbol_centers=None):
    # function that interpolates to the constellation
    # assumes xhat and syms are 1D arrays
    if symbol_centers is None:
        symbol_centers = syms
    absdiff = np.abs(xhat[:, np.newaxis] - symbol_centers[np.newaxis, :])
    min_indices = np.argmin(absdiff, axis=1)
    assert (len(min_indices) == len(xhat))
    return syms[min_indices]


def calculate_mmse_weights(transfer_function, num_taps, snr_db, ref_tap=0, input_delay=0):
    # Routine ported from MATLABs LinearEqualizer library
    # Details in: https://www2.ece.ohio-state.edu/~schniter/ee501/handouts/mmse_eq.pdf
    total_delay = ref_tap + input_delay
    assert (len(transfer_function) <= num_taps)
    h_len = len(transfer_function)
    h_auto_corr = np.correlate(transfer_function, transfer_function, mode='full')
    r = np.concatenate((h_auto_corr[h_len - 1:], np.zeros((num_taps - h_len,))))
    R = toeplitz(r, np.conj(r))
    R = R + 0.5 * 10**(-snr_db / 10) * np.eye(num_taps)
    p = np.zeros((num_taps,), dtype=transfer_function.dtype)
    if total_delay < h_len:
        p[0:total_delay] = np.conjugate(np.flipud(transfer_function[0:total_delay]))
    else:
        idxs = slice(total_delay, total_delay - h_len, -1)
        p[idxs] = np.conjugate(transfer_function)
    w0 = np.linalg.solve(R, p)
    return w0


def calculate_weight_error(hhat, hopt, ref_tap=0):
    ref_tap_angle = np.angle(hhat[ref_tap])
    weight_error = np.sum(np.square(np.absolute(hopt - np.exp(-1j * ref_tap_angle) * hhat)))
    return weight_error


def calculate_confusion_matrix(ahat, a):
    assert (np.all(np.unique(ahat) == np.unique(a)))
    unique_symbols, true_symbols = np.unique(a, return_inverse=True)
    unique_symbols2, pred_symbols = np.unique(ahat, return_inverse=True)

    N = len(unique_symbols)
    conf_mat = np.zeros((N, N), dtype=np.int16)

    for true_label, predicted_label in zip(true_symbols, pred_symbols):
        conf_mat[true_label, predicted_label] += 1

    return conf_mat, unique_symbols, unique_symbols2
