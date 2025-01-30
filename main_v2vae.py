"""
    Main script for fitting (blind) equalizers to a non-linear
    Wiener-Hammestein channel (with polynomial non-linearity)
"""

import time
import komm
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.data_generation import NonLinearISI
from lib.real_equalization import SecondVolterraVAE, SecondVolterraV2VAE, SecondVolterraPilot
from lib.utility import calc_ser_pam, calc_theory_ser_pam, calc_ser_from_probs


if __name__ == "__main__":
    # Parameters to be used
    samples_per_symbol_in = 4
    samples_per_symbol_out = 2
    seed = 124545
    snr_db = 16.0
    N_symbols = int(1e6)
    N_symbols_val = int(1e6)  # number of symbols used for SER calculation
    eq_lags1 = 25
    eq_lags2 = 25
    channel_memory = 15

    # Create modulation scheme
    order = 4
    modulation_scheme = komm.PAModulation(order=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    constellation = modulation_scheme.constellation
    print(f"Constellation of order {order} is: {constellation}")
    print(f"Avg. symbol energy: {symbol_energy}")

    # Create random object
    random_obj = np.random.default_rng(0)

    # Create data generation object
    wh_config = {
        "lp1_config": {
            "order": 5,
            "cutoff": 0.65,
            "lp_type": "bessel"
        },
        "lp2_config": {
            "order": 5,
            "cutoff": 0.55,
            "lp_type": "bessel"
        },
        "nl_type": 'eam',
        "laser_power_dbm": 0.0,
        "dac_vpp": 4.0,
        "dac_vb": -2.0
    }

    nonlinisi = NonLinearISI(oversampling=samples_per_symbol_in,
                                wh_config=wh_config,
                                wh_type='lp',
                                snr_db=snr_db,
                                samples_pr_symbol=samples_per_symbol_out,
                                constellation=constellation,
                                random_obj=random_obj,
                                rrc_length=samples_per_symbol_in * 8 - 1,
                                rrc_rolloff=0.1)

    # Generate training data
    rx, syms = nonlinisi.generate_data(N_symbols)
    rx_val, syms_val = nonlinisi.generate_data(N_symbols)
    EsN0_db = nonlinisi.EsN0_db


    # Create VAE object and process samples
    vol2_v2vae = SecondVolterraV2VAE(
        channel_memory=channel_memory,
        equaliser_n_lags1=eq_lags1,
        equaliser_n_lags2=eq_lags2,
        learning_rate=5e-3,
        constellation=constellation,
        batch_size=500,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
        dtype=torch.float32
    )
    print(f"{vol2_v2vae} is probabilistic: {vol2_v2vae.IS_PROBABILISTIC}")
    start_time = time.time()
    vol2_v2vae.initialize_optimizer()
    __ = vol2_v2vae.fit(rx)
    q_vol2_v2vae = vol2_v2vae.estimate_symbol_probs(rx_val)
    y_vol2_v2vae = np.sum(q_vol2_v2vae * vol2_v2vae.constellation.numpy()[:, None], axis=0)
    print(f"Elapsed time: {time.time() - start_time}")

    # Create the Linear VAE object and process
    vol2_vae = SecondVolterraVAE(
        channel_memory=channel_memory,
        equaliser_n_lags1=eq_lags1,
        equaliser_n_lags2=eq_lags2,
        learning_rate=5e-3,
        constellation=constellation,
        batch_size=500,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
        dtype=torch.float32
    )
    print(f"{vol2_vae} is probabilistic: {vol2_vae.IS_PROBABILISTIC}")
    start_time = time.time()
    vol2_vae.initialize_optimizer()
    __ = vol2_vae.fit(rx)
    q_vol2_vae = vol2_vae.estimate_symbol_probs(rx_val)
    y_vol2_vae = np.sum(q_vol2_vae * vol2_vae.constellation.numpy()[:, None], axis=0)
    print(f"Elapsed time: {time.time() - start_time}")

    # Run supervised Volterra series as comparison
    mse_pilot = SecondVolterraPilot(n_lags1=eq_lags1, n_lags2=eq_lags2, learning_rate=5e-3,
                                    samples_per_symbol=samples_per_symbol_out, batch_size=400,
                                    dtype=torch.float32, 
                                    torch_device=torch.device("cpu"))
    print(f"{mse_pilot} is probabilistic: {mse_pilot.IS_PROBABILISTIC}")
    start_time = time.time()
    mse_pilot.initialize_optimizer()
    __ = mse_pilot.fit(rx, syms)
    y_eq_mse = mse_pilot.apply(rx_val)
    print(f"Elapsed time: {time.time() - start_time}")

    # Make "constellation plot" - noisy symbol + equalized symbol
    fig, ax = plt.subplots(ncols=2, nrows=2)
    nbins = 100
    for this_ax, yout, label in zip(ax.flatten(),
                                    (rx_val[::samples_per_symbol_out], y_vol2_v2vae, y_vol2_vae, y_eq_mse),
                                    ('Noisy symbols', 'V2VAE', 'LinVAE', f"{mse_pilot}")):
        this_ax.hist(yout, bins=nbins)
        this_ax.set_title(label)

    # Calculate error metrics - Symbol Error Rate (SER)
    ser_v2vae, __ = calc_ser_from_probs(q_vol2_v2vae, syms_val, vol2_v2vae.constellation.numpy())
    ser_vae, __ = calc_ser_from_probs(q_vol2_vae, syms_val, vol2_vae.constellation.numpy())
    ser_mse, __ = calc_ser_pam(y_eq_mse, syms_val)
    ser_no_eq, __ = calc_ser_pam(rx_val[::samples_per_symbol_out], syms_val)
    ser_theory = calc_theory_ser_pam(order, EsN0_db)

    for ser, method in zip([ser_v2vae, ser_vae, ser_mse, ser_no_eq, ser_theory],
                           ["V2VAE", "VAE", f"{mse_pilot}", "No eq", "Theory"]):
        print(f"{method}: {ser:.4e} (SER)")

    # Plot non-linearity in Wiener-Hammerstein system
    fig, ax = plt.subplots()
    x = np.linspace(np.min(constellation), np.max(constellation), 1000)
    ax.plot(x, nonlinisi.wh.nl(x))
    ax.set_title('Non-linearity in WH')

    plt.show()
