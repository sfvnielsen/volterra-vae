
# Module that encompasses data generation for these digital communication systems
import numpy as np
import komm
from .utility import symbol_sync, find_max_variance_sample
# from optic.core import parameters
# from optic.dsp.core import decimate, symbolSync


# FIXME: Handle normalization of constellation in a principled way throughout the framework


def generate_symbol_data_with_isi(random_obj: np.random.Generator, snr_db, modulation_scheme: komm.Modulation, n_symbols, h_isi):
    """ Generate symbol-level data from the AWGN channel with linear intersymbol-interference
        Blockdiagram:
        bits -> symbols -> ISI -> add white noise
    """
    n_symbols_conv = n_symbols + len(h_isi)  # extra symbols to pad with for ISI response
    n_bits = int(n_symbols_conv * np.log2(len(modulation_scheme.constellation)))
    bit_sequence = random_obj.integers(0, 2, size=n_bits)
    a = modulation_scheme.modulate(bit_sequence)  # complex symbol sequence

    # Construct Tx signal (simple 1 symbol pr sample) and add noise
    x = np.convolve(a, h_isi, mode='valid')
    noise_std = np.sqrt(np.average(np.square(np.abs(modulation_scheme.constellation))) / (2 * 10**(snr_db / 10)))
    awgn = noise_std * random_obj.standard_normal(len(x))
    if np.iscomplexobj(x):
        awgn = awgn.astype(np.complex128)
        awgn += 1j * noise_std * random_obj.standard_normal(len(x))
    rx = x + awgn
    rx = rx[0:n_symbols]  # truncate to valid symbols
    a = a[(len(h_isi) - 1):(n_symbols + len(h_isi) - 1)]

    # Calculate resulting EsN0
    EsN0_db = 10.0 * np.log10(1.0 / noise_std ** 2)

    return rx, a, modulation_scheme.constellation, EsN0_db


def generate_data_with_pulse_shaping(random_obj: np.random.Generator, snr_db, modulation_scheme: komm.Modulation,
                                     pulse_shaping_filter, n_symbols, sps_in, sps_out):
    """ Generate data from the AWGN channel with matched filtering and decimation
        Blockdiagram:
        bits -> symbols -> upsampling -> pulse-shaping -> add white noise -> matched filtering -> decimation
        # FIXME: Make this a special case of non-linear function with f(x) = x
    """
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)

    constellation = modulation_scheme.constellation
    a = modulation_scheme.modulate(bit_sequence)

    # Upsample symbol sequence and pulse shape
    a_zeropadded = np.zeros(sps_in * len(a), dtype=a.dtype)
    a_zeropadded[0::sps_in] = a
    x = np.convolve(a_zeropadded, pulse_shaping_filter)
    gg = np.convolve(pulse_shaping_filter, pulse_shaping_filter[::-1])
    pulse_energy = np.max(gg)
    sync_point = np.argmax(gg)

    # AWGN channel
    noise_std = np.sqrt(np.average(np.square(np.abs(constellation))) * pulse_energy / (2 * 10 ** (snr_db / 10)))
    rx = x + noise_std * random_obj.standard_normal(len(x))
    if np.iscomplexobj(constellation):
        rx += 1j * noise_std * random_obj.standard_normal(len(x))

    # Apply matched filter
    rx = np.convolve(rx, pulse_shaping_filter[::-1])[sync_point:-sync_point] / pulse_energy

    # TODO: Remove dependency to optic
    # Decimate if sps_in != sps_out
    if sps_in != sps_out:
        dec_pars = parameters()
        dec_pars.SpS_in = sps_in
        dec_pars.SpS_out = sps_out
        rx = np.squeeze(decimate(rx[:, np.newaxis], dec_pars))

    # Calculate resulting EsN0
    EsN0_db = 10.0 * np.log10(pulse_energy / noise_std ** 2)

    return rx, a, constellation, EsN0_db


def generate_data_pulse_shaping_linear_isi(random_obj: np.random.Generator, snr_db, modulation_scheme: komm.Modulation,
                                           pulse_shaping_filter, n_symbols, sps_in, sps_out, h_isi):
    """ Generate data from the AWGN channel with matched filtering and decimation
        Inter-symbol-interference modeled as a FIR filter
        ISI transfer function taken from (Caciularu, 2020)
        Blockdiagram:
        bits -> symbols -> upsampling -> pulse-shaping -> ISI convolution -> add white noise -> matched filtering -> decimation
    """

    n_extra_symbols = len(h_isi)  # due to isi reponse
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * (n_symbols + n_extra_symbols))
    bit_sequence = random_obj.integers(0, 2, size=n_bits)

    constellation = modulation_scheme.constellation
    a = modulation_scheme.modulate(bit_sequence)
    #constellation /= np.sqrt(np.average(np.absolute(constellation)**2))
    #a = modulation_scheme.modulate(bit_sequence) / np.sqrt(np.average(np.absolute(constellation)**2))

    # Upsample symbol sequence and pulse shape
    a_zeropadded = np.zeros(sps_in * len(a), dtype=a.dtype)
    a_zeropadded[0::sps_in] = a
    x = np.convolve(a_zeropadded, pulse_shaping_filter)
    gg = np.convolve(pulse_shaping_filter, pulse_shaping_filter[::-1])
    pulse_energy = np.max(gg)
    sync_point = np.argmax(gg)

    # Convolve with ISI transfer function
    h_isi_zeropadded = np.zeros(sps_in * (len(h_isi) - 1) + 1, dtype=h_isi.dtype)
    h_isi_zeropadded[::sps_in] = h_isi
    h_isi_zeropadded = h_isi_zeropadded / np.linalg.norm(h_isi_zeropadded)
    x = np.convolve(x, h_isi_zeropadded)

    # AWGN channel
    noise_std = np.sqrt(np.average(np.square(np.abs(constellation))) * pulse_energy / (2 * 10 ** (snr_db / 10)))
    rx = x + noise_std * random_obj.standard_normal(len(x))
    if np.iscomplexobj(rx):
        rx += 1j * noise_std * random_obj.standard_normal(len(x))

    # Apply matched filter, sample recovery and decimation
    rx = np.convolve(rx, pulse_shaping_filter[::-1]) / pulse_energy
    rx = rx[sync_point:-sync_point]
    max_var_samp = find_max_variance_sample(rx, sps_in)
    decimation_factor = int(sps_in / sps_out)
    assert sps_in % sps_out == 0
    rx = np.roll(rx, -max_var_samp)[::decimation_factor]

    # Sync symbols
    a = symbol_sync(rx, a, sps_out)[0:n_symbols]
    rx = rx[0:n_symbols*sps_out]

    # Calculate resulting EsN0
    EsN0_db = 10.0 * np.log10(pulse_energy / noise_std ** 2)

    return rx, a, constellation, EsN0_db


def generate_data_nonlin_simple(random_obj: np.random.Generator, snr_db, modulation_scheme: komm.Modulation,
                                pulse_shaping_filter, n_symbols, sps_in, sps_out, linear_isi, non_lin_coef):
    """ Generate data from the AWGN channel with a non-linearity after pulse shaping
        Blockdiagram:
        bits -> symbols -> upsampling -> pulse-shaping -> apply non-linearity (isi + square) -> add white noise -> decimate
    """
    # FIXME: Check this function.

    ps_filter_len_in_syms = len(pulse_shaping_filter) // sps_in
    n_extra_symbols = 2 * len(linear_isi) + ps_filter_len_in_syms
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * (n_symbols + n_extra_symbols))
    bit_sequence = random_obj.integers(0, 2, size=n_bits)

    constellation = modulation_scheme.constellation
    a = modulation_scheme.modulate(bit_sequence)

    # Upsample symbol sequence and pulse shape
    a_zeropadded = np.zeros(sps_in * (n_symbols + n_extra_symbols), dtype=a.dtype)
    a_zeropadded[0::sps_in] = a
    x = np.convolve(a_zeropadded, pulse_shaping_filter)
    gg = np.convolve(pulse_shaping_filter, pulse_shaping_filter[::-1])
    pulse_energy = np.max(gg)
    sync_point = np.argmax(gg)

    # AWGN channel with non-linearity (FIR + squaring + FIR)
    h_isi_zeropadded = np.zeros(sps_in * (len(linear_isi) - 1) + 1, dtype=linear_isi.dtype)
    h_isi_zeropadded[::sps_in] = linear_isi
    h_isi_zeropadded = h_isi_zeropadded / np.linalg.norm(h_isi_zeropadded)
    rx = np.convolve(h_isi_zeropadded, x)
    rx = (1 - non_lin_coef) * rx + non_lin_coef * rx**2
    rx = np.convolve(h_isi_zeropadded, rx)

    # Calculate empirical energy pr. symbol period
    Es_emp = np.mean(np.sum(np.square(np.reshape(rx[0:n_symbols * sps_in], (-1, sps_in))), axis=1))
    
    # Converting average energy pr. symbol into base power (cf. https://wirelesspi.com/pulse-amplitude-modulation-pam/)
    base_power_pam = Es_emp * (3 / (len(constellation)**2 - 1))

    # Derive noise std
    noise_std = np.sqrt(base_power_pam / 10.0 **(snr_db / 10.0))
    #noise_std = np.sqrt(np.average(np.square(np.abs(constellation))) * pulse_energy / (2 * 10 ** (snr_db / 10)))
    rx += noise_std * random_obj.standard_normal(len(rx))
    if np.iscomplexobj(constellation):
        rx += 1j * noise_std * random_obj.standard_normal(len(rx))

    # Matched filter, sample recovery and decimation
    rx = np.convolve(rx, pulse_shaping_filter[::-1]) / pulse_energy
    rx = rx[sync_point:-sync_point]
    max_var_samp = find_max_variance_sample(rx, sps_in)
    decimation_factor = int(sps_in / sps_out)
    assert sps_in % sps_out == 0
    rx = np.roll(rx, -max_var_samp)[::decimation_factor]

    # Sync symbols
    a = symbol_sync(rx, a, sps_out)[0:n_symbols]
    rx = rx[0:n_symbols*sps_out]

    # Calulate pr. symbol SNR
    EsN0_db = snr_db  # due to above adjustment EsN0_db == snr_db

    return rx, a, constellation, EsN0_db
