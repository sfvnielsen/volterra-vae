import torch
import numpy as np
import komm
import matplotlib.pyplot as plt
from lib.complex_equalization import LinearVAE
from lib.utility import calc_brute_force_ser
from lib.data_generation import generate_data_pulse_shaping_linear_isi, generate_symbol_data_with_isi
from commpy.filters import rrcosfilter


# FIXME FIXME!!!! 
# Works in Lauinger repo. So something is off with the data generation?
# What is up with this weird streching of the constellation?


if __name__ == "__main__":
    # Parameters to be used
    num_eq_taps = 25
    #samples_per_symbol_in = 16
    samples_per_symbol_out = 1
    seed = 124545
    snr_db = 20.0

    # Artificial transfer function of the channel
    h_orig = np.array([0.0545 + 1j * 0.05, 0.2823 - 1j * 0.11971, -0.7676 + 1j * 0.2788, -0.0641 - 1j * 0.0576, 0.0466 - 1j * 0.02275])  # From Caciularu 2020

    # Create modulation scheme
    N_symbols = int(10e5)
    order = 16
    modulation_scheme = komm.QAModulation(orders=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    const_normalization = np.sqrt(np.mean(np.abs(modulation_scheme.constellation)**2))
    constellation = modulation_scheme.constellation / const_normalization
    print(f'Constellation of order {order} is: {constellation}')
    print(f'Avg. symbol energy: {symbol_energy}')

    """
    # Create pulse shape
    sym_rate = 10e6
    sym_length = 1 / sym_rate
    Ts = sym_length / samples_per_symbol_in  # effective sampling interval
    pulse_length_in_symbols = 32
    rolloff = 0.1
    t, g = rrcosfilter(pulse_length_in_symbols * samples_per_symbol_in, rolloff, sym_length, 1 / Ts)
    g = g / np.linalg.norm(g)
    """

    # Generate data
    random_obj = np.random.default_rng(seed)
    rx, syms, constellation, __ = generate_symbol_data_with_isi(random_obj, snr_db, modulation_scheme, N_symbols, h_orig)

    # Create VAE object and process samples
    lin_vae = LinearVAE(encoder_n_taps=num_eq_taps, decoder_n_taps=num_eq_taps, learning_rate=5 * 1e-3, constellation=constellation,
                        batch_size=400, samples_per_symbol=samples_per_symbol_out, noise_variance=10**(-snr_db / 10),
                        adaptive_noise_variance=False, torch_device=torch.device('cpu'))
    y_eq = lin_vae.fit(rx)
    y_eq = y_eq / np.sqrt(np.average(np.absolute(y_eq) ** 2)) * np.sqrt(np.average(np.absolute(constellation) ** 2))

    # Make "constellation plot" - noisy symbol + equalized symbol
    decimation_factor = 100
    fig, ax = plt.subplots()
    ax.plot(np.real(rx[::decimation_factor * samples_per_symbol_out]), np.imag(rx[::decimation_factor * samples_per_symbol_out]), 'rx', label='Noisy symbols')
    ax.plot(np.real(y_eq[::decimation_factor]), np.imag(y_eq[::decimation_factor]), 'b^', label='VAE')
    ax.grid()
    ax.legend()

    # Calculate error metrics - Symbol Error Rate (SER)
    ser_vae, __, __ = calc_brute_force_ser(y_eq, syms, delay_order=5)
    ser_no_eq, __, __ = calc_brute_force_ser(rx[::samples_per_symbol_out], syms, delay_order=5)

    for ser, method in zip([ser_vae, ser_no_eq],
                           ['VAE', 'No eq']):
        print(f"{method}: {ser} (SER)")

    plt.show()
