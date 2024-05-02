import komm
import matplotlib.pyplot as plt
import numpy as np
import torch
from commpy.filters import rrcosfilter

from lib.complex_equalization import LinearVAE
from lib.data_generation import SymbolAWGNwithISI
from lib.utility import calc_brute_force_ser

if __name__ == "__main__":
    # Parameters to be used in this simulation
    num_eq_taps = 25
    samples_per_symbol_in = 16
    samples_per_symbol_out = 2
    seed = 124545
    snr_db = 20.0

    # Artificial transfer function of the channel
    h_orig = np.array(
        [0.0545 + 1j * 0.05, 0.2823 - 1j * 0.11971, -0.7676 + 1j * 0.2788,
            -0.0641 - 1j * 0.0576, 0.0466 - 1j * 0.02275])  # From Caciularu 2020

    # Create modulation scheme
    N_symbols = int(1e6)
    N_symbols_val = int(1e6)
    order = 16
    modulation_scheme = komm.QAModulation(orders=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    constellation = modulation_scheme.constellation
    print(f"Constellation of order {order} is: {constellation}")
    print(f"Avg. symbol energy: {symbol_energy}")


    # Generate data
    random_obj = np.random.default_rng(seed)
    symbol_awgn = SymbolAWGNwithISI(h_isi=h_orig,
                                    snr_db=snr_db,
                                    constellation=constellation,
                                    random_obj=random_obj)
    rx, syms = symbol_awgn.generate_data(N_symbols)
    rx_val, syms_val = symbol_awgn.generate_data(N_symbols_val)

    # Create VAE object and process samples
    lin_vae = LinearVAE(
        encoder_n_taps=num_eq_taps,
        decoder_n_taps=num_eq_taps,
        learning_rate=5 * 1e-3,
        constellation=constellation,
        batch_size=400,
        samples_per_symbol=1,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
    )
    y_eq = lin_vae.fit(rx)
    y_eq = lin_vae.apply(rx_val)

    # Make "constellation plot" - noisy symbol + equalized symbol
    decimation_factor = 100
    fig, ax = plt.subplots()
    ax.plot(np.real(rx_val[:: decimation_factor]), np.imag(rx_val[:: decimation_factor]), "rx", label="Noisy symbols")
    ax.plot(np.real(y_eq[::decimation_factor]), np.imag(y_eq[::decimation_factor]), "b^", label="VAE")
    ax.grid()
    ax.legend()

    # Calculate error metrics - Symbol Error Rate (SER)
    ser_vae, __, __ = calc_brute_force_ser(y_eq, syms_val, delay_order=5)
    ser_no_eq, __, __ = calc_brute_force_ser(rx_val, syms_val, delay_order=5)

    for ser, method in zip([ser_vae, ser_no_eq], ["VAE", "No eq"]):
        print(f"{method}: {ser} (SER)")

    plt.show()
