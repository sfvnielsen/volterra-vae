import torch
import numpy as np
import komm
import matplotlib.pyplot as plt
from lib.real_equalization import LinearVAE, LMSPilot
from lib.utility import calc_ser_pam
from lib.data_generation import generate_data_pulse_shaping_linear_isi
from commpy.filters import rrcosfilter


if __name__ == "__main__":
    # Parameters to be used
    num_eq_taps = 25
    samples_per_symbol_in = 16
    samples_per_symbol_out = 2
    seed = 124545
    snr_db = 12.0

    # Artificial transfer function of the channel
    h_orig = np.array([0.2, 0.9, 0.3])  # From Caciularu 2020
    #h_orig = np.array([1.0, 0.0, 0.0])  # debug

    # Create modulation scheme
    N_symbols = int(10e5)
    order = 4
    modulation_scheme = komm.PAModulation(order=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    const_normalization = np.sqrt(np.mean(np.abs(modulation_scheme.constellation)**2))
    constellation = modulation_scheme.constellation / const_normalization
    print(f'Constellation of order {order} is: {constellation}')
    print(f'Avg. symbol energy: {symbol_energy}')

    # Create pulse shape
    sym_rate = 10e6
    sym_length = 1 / sym_rate
    Ts = sym_length / samples_per_symbol_in  # effective sampling interval
    pulse_length_in_symbols = 32
    rolloff = 0.1
    t, g = rrcosfilter(pulse_length_in_symbols * samples_per_symbol_in, rolloff, sym_length, 1 / Ts)
    g = g / np.linalg.norm(g)

    # Generate data
    random_obj = np.random.default_rng(seed)
    rx, syms, constellation, __ = generate_data_pulse_shaping_linear_isi(random_obj, snr_db, modulation_scheme, g, N_symbols, samples_per_symbol_in, samples_per_symbol_out, h_orig)

    # Create VAE object and process samples
    lin_vae = LinearVAE(channel_n_taps=num_eq_taps, equaliser_n_taps=num_eq_taps, learning_rate=5 * 1e-3, constellation=constellation,
                        batch_size=400, samples_per_symbol=samples_per_symbol_out, noise_variance=10**(-snr_db / 10),
                        adaptive_noise_variance=True, torch_device=torch.device('cpu'))
    y_eq = lin_vae.fit(rx)
    y_eq = y_eq / np.sqrt(np.average(np.absolute(y_eq) ** 2)) * np.sqrt(np.average(np.absolute(constellation) ** 2))

    # Run LMS with pilots as a comparison
    lms = LMSPilot(n_taps=num_eq_taps, learning_rate=1e-4, samples_per_symbol=samples_per_symbol_out)
    y_eq_lms = lms.fit(rx, syms)

    # Make "constellation plot" - noisy symbol + equalized symbol
    fig, ax = plt.subplots()
    ax.hist(rx[::samples_per_symbol_out], bins=100, label='Noisy symbols')
    ax.hist(y_eq, bins=100, label='VAE')
    ax.hist(y_eq_lms, bins=100, label='LMSPilot')
    ax.grid()
    ax.legend()

    # Calculate error metrics - Symbol Error Rate (SER) (only on the last half of the data)
    ser_vae, __ = calc_ser_pam(y_eq[N_symbols//2:], syms[N_symbols//2:], delay_order=5)
    ser_lms, __ = calc_ser_pam(y_eq_lms[N_symbols//2:], syms[N_symbols//2:], delay_order=5)
    ser_no_eq, __ = calc_ser_pam(rx[::samples_per_symbol_out][N_symbols//2:], syms[N_symbols//2:], delay_order=5)

    for ser, method in zip([ser_vae, ser_lms, ser_no_eq],
                           ['VAE', 'LMS', 'No eq']):
        print(f"{method}: {ser} (SER)")

    plt.show()
