
# Module that encompasses data generation for these digital communication systems
import numpy as np
import komm
from optic.core import parameters
from optic.dsp.core import decimate


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
    #a = modulation_scheme.modulate(bit_sequence)
    constellation /= np.sqrt(np.average(np.absolute(constellation)**2))
    a = modulation_scheme.modulate(bit_sequence) / np.sqrt(np.average(np.absolute(constellation)**2))

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

    # Apply matched filter and sync
    # FIXME: Describe the syncing af Rx and syms here!
    rx = np.convolve(rx, pulse_shaping_filter[::-1]) / pulse_energy
    rx = rx[((n_extra_symbols - 1) * sps_in + sync_point):((n_extra_symbols + n_symbols - 1) * sps_in + sync_point)]

    # Decimate if sps_in != sps_out
    if sps_in != sps_out:
        dec_pars = parameters()
        dec_pars.SpS_in = sps_in
        dec_pars.SpS_out = sps_out
        rx = np.squeeze(decimate(rx[:, np.newaxis], dec_pars))

    # Calculate resulting EsN0
    EsN0_db = 10.0 * np.log10(pulse_energy / noise_std ** 2)

    return rx, a[n_extra_symbols:], constellation, EsN0_db


def generate_data_nonlin_simple(random_obj: np.random.Generator, snr_db, modulation_scheme: komm.Modulation,
                                pulse_shaping_filter, n_symbols, sps_in, sps_out, non_linear_function=lambda xin: xin):
    """ Generate data from the AWGN channel with a non-linearity after pulse shaping
        Blockdiagram:
        bits -> symbols -> upsampling -> pulse-shaping -> apply non-linearity -> add white noise -> matched filtering -> decimation
    """
    n_bits = int(np.log2(len(modulation_scheme.constellation)) * n_symbols)
    bit_sequence = random_obj.integers(0, 2, size=n_bits)

    constellation = modulation_scheme.constellation
    a = modulation_scheme.modulate(bit_sequence)

    constellation_normalizer = np.sqrt(np.mean(np.absolute(constellation)**2))
    constellation /= constellation_normalizer
    a /= constellation_normalizer

    # Upsample symbol sequence and pulse shape
    a_zeropadded = np.zeros(sps_in * len(a), dtype=a.dtype)
    a_zeropadded[0::sps_in] = a
    x = np.convolve(a_zeropadded, pulse_shaping_filter)
    gg = np.convolve(pulse_shaping_filter, pulse_shaping_filter[::-1])
    pulse_energy = np.max(gg)
    sync_point = np.argmax(gg)

    # AWGN channel with non-linearity
    rx = non_linear_function(x)

    # Normalize and calculate a noise level based on desired SNR (after non-linearity)
    rx = (rx - np.mean(rx)) / np.std(rx)
    rx_syms_noise_free = np.convolve(rx, pulse_shaping_filter[::-1])[sync_point:-sync_point:sps_in] / pulse_energy

    noise_std = np.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))

    # Add white noise with at desired lvl
    rx += noise_std * random_obj.standard_normal(len(x))
    if np.iscomplexobj(constellation):
        rx += 1j * noise_std * random_obj.standard_normal(len(x))

    # Apply matched filter
    rx = np.convolve(rx, pulse_shaping_filter[::-1])[sync_point:-sync_point] / pulse_energy

    # Decimate if sps_in != sps_out
    if sps_in != sps_out:
        dec_pars = parameters()
        dec_pars.SpS_in = sps_in
        dec_pars.SpS_out = sps_out
        rx = np.squeeze(decimate(rx[:, np.newaxis], dec_pars))

    # Calculate symbol energy (diff between peaks) - used for theoretical calculation of SER
    average_symbol_energy = 0.0
    constellation_points = np.unique(np.real(constellation))
    for (i,j) in zip(range(0, len(constellation_points) - 1), range(1, len(constellation_points))):
        idx_i, idx_j = np.where(np.real(a) == constellation_points[i]), np.where(np.real(a) == constellation_points[j])
        average_symbol_energy += np.square((np.average(np.real(rx_syms_noise_free[idx_i])) - np.average(np.real(rx_syms_noise_free[idx_j]))) / 2)
    average_symbol_energy /= len(constellation_points) - 1

    symbol_snr = average_symbol_energy / (noise_std**2)
    EsN0_db = 10.0 * np.log10(symbol_snr)

    return rx, a, constellation, EsN0_db
