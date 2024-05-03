import komm
import matplotlib.pyplot as plt
import numpy as np
import torch
from commpy.filters import rrcosfilter

from lib.data_generation import NonLinearISI
from lib.real_equalization import SecondVolterraVAE, LMSPilot, SecondVolterraVOLVO, SecondVolterraPilot, TorchLMSPilot
from lib.utility import calc_ser_pam, calc_theory_ser_pam

if __name__ == "__main__":
    # Parameters to be used
    num_eq_taps = 25
    samples_per_symbol_in = 8
    samples_per_symbol_out = 2
    seed = 124545
    snr_db = 12.0
    N_symbols = int(1e6)
    N_symbols_val = int(1e6)  # number of symbols used for SER calculation

    # Inter-symbol-interference transfer function
    # h_orig = np.array([0.2, 0.9, 0.3])  # From Caciularu 2020 - NB! Non-minimum phase
    h_orig = np.array([1.0, 0.3, 0.1])  # simple minimum phase with zeros at (0.2, -0.5)

    # Create modulation scheme
    order = 4
    modulation_scheme = komm.PAModulation(order=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    constellation = modulation_scheme.constellation
    print(f"Constellation of order {order} is: {constellation}")
    print(f"Avg. symbol energy: {symbol_energy}")

    # Create random object
    random_obj = np.random.default_rng(12345)

    # Create data generation object
    wh_config = {
        "fir1": h_orig,
        "fir2": h_orig,
        "poly_coefs": [0.9, 0.1, 0.0]
    }
    psawgn = NonLinearISI(oversampling=samples_per_symbol_in,
                          wh_config=wh_config,
                          snr_db=snr_db,
                          samples_pr_symbol=samples_per_symbol_out,
                          constellation=constellation,
                          random_obj=random_obj,
                          rrc_length=samples_per_symbol_in * 8 - 1,
                          rrc_rolloff=0.1)

    # Generate training data
    rx, syms = psawgn.generate_data(N_symbols)
    rx_val, syms_val = psawgn.generate_data(N_symbols)
    EsN0_db = psawgn.EsN0_db


    # Create VAE object and process samples
    channel_memory = 9
    vol2_volvo = SecondVolterraVOLVO(
        channel_memory=channel_memory,
        equaliser_n_lags1=num_eq_taps,
        equaliser_n_lags2=5,
        learning_rate=5e-3,
        constellation=constellation,
        batch_size=400,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
        dtype=torch.float32
    )
    vol2_volvo.initialize_optimizer()
    __ = vol2_volvo.fit(rx)
    y_vol2_volvo = vol2_volvo.apply(rx_val)

    # Create the Linear VAE object and process
    vol2_vae = SecondVolterraVAE(
        channel_memory=channel_memory,
        equaliser_n_lags1=num_eq_taps,
        equaliser_n_lags2=5,
        learning_rate=5e-3,
        constellation=constellation,
        batch_size=400,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
        dtype=torch.float32
    )
    vol2_vae.initialize_optimizer()
    __ = vol2_vae.fit(rx)
    y_vol2_vae = vol2_vae.apply(rx_val)

    # Run LMS with pilots as a comparison
    #mse_pilot = TorchLMSPilot(n_taps=num_eq_taps, reference_tap=num_eq_taps//2,
    #                          learning_rate=5e-4,
    #                          samples_per_symbol=samples_per_symbol_out, batch_size=400,
    #                          dtype=torch.float32, 
    #                          torch_device=torch.device("cpu"))
    #mse_pilot = LMSPilot(
    #    n_taps=num_eq_taps,
    #    learning_rate=1e-4,
    #    samples_per_symbol=samples_per_symbol_out,
    #)
    mse_pilot = SecondVolterraPilot(n_lags1=num_eq_taps, n_lags2=9, learning_rate=5e-3,
                                    samples_per_symbol=samples_per_symbol_out, batch_size=400,
                                    dtype=torch.float32, 
                                    torch_device=torch.device("cpu"))
    mse_pilot.initialize_optimizer()
    __ = mse_pilot.fit(rx, syms)
    y_eq_mse = mse_pilot.apply(rx_val)

    # Make "constellation plot" - noisy symbol + equalized symbol
    fig, ax = plt.subplots(ncols=2, nrows=2)
    nbins = 100
    for this_ax, yout, label in zip(ax.flatten(),
                                    (rx_val[::samples_per_symbol_out], y_vol2_volvo, y_vol2_vae, y_eq_mse),
                                    ('Noisy symbols', 'VOLVO', 'LinVAE', f"{mse_pilot}")):
        this_ax.hist(yout, bins=nbins)
        this_ax.set_title(label)

    # Calculate error metrics - Symbol Error Rate (SER)
    ser_volvo, __ = calc_ser_pam(y_vol2_volvo, syms_val)
    ser_vae, __ = calc_ser_pam(y_vol2_vae, syms_val)
    ser_mse, __ = calc_ser_pam(y_eq_mse, syms_val)
    ser_no_eq, __ = calc_ser_pam(rx_val[::samples_per_symbol_out], syms_val)
    ser_theory = calc_theory_ser_pam(order, EsN0_db)

    for ser, method in zip([ser_volvo, ser_vae, ser_mse, ser_no_eq, ser_theory],
                           ["VOLVO", "VAE", f"{mse_pilot}", "No eq", "Theory"]):
        print(f"{method}: {ser:.4e} (SER)")

    plt.show()
