import komm
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.data_generation import PulseShapedAWGN
from lib.real_equalization import LinearVAE, LMSPilot, ConvolutionalNNPilot
from lib.utility import calc_ser_pam, calc_theory_ser_pam


if __name__ == "__main__":
    # Parameters to be used
    num_eq_taps = 25
    samples_per_symbol_in = 8
    samples_per_symbol_out = 2
    seed = 124545
    snr_db = 12.0
    N_symbols = int(2.5e6)
    N_symbols_val = int(1e6)  # number of symbols used for SER calculation

    # Artificial transfer function of the channel
    # h_orig = np.array([0.2, 0.9, 0.3])  # From Caciularu 2020
    h_orig = np.array([1.0, 0.3, 0.1])  # simple minimum phase with zeros at (0.2, -0.5)

    # Create modulation scheme
    order = 4
    modulation_scheme = komm.PAModulation(order=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    constellation = modulation_scheme.constellation
    print(f"Constellation of order {order} is: {constellation}")
    print(f"Avg. symbol energy: {symbol_energy}")

    # Generate data - training and validation
    random_obj = np.random.default_rng(seed)

    # Create data generation object
    psawgn = PulseShapedAWGN(oversampling=samples_per_symbol_in,
                             h_isi=h_orig,
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
    lin_vae = LinearVAE(
        channel_n_taps=num_eq_taps,
        equaliser_n_taps=num_eq_taps,
        learning_rate=5 * 1e-3,
        constellation=constellation,
        batch_size=400,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
    )
    lin_vae.initialize_optimizer()
    __ = lin_vae.fit(rx)
    y_eq = lin_vae.apply(rx_val)

    # Run LMS with pilots as a comparison
    lms = LMSPilot(
        n_taps=num_eq_taps,
        learning_rate=5e-5,
        samples_per_symbol=samples_per_symbol_out,
    )
    __ = lms.fit(rx, syms)
    y_eq_lms = lms.apply(rx_val)

    # Compare to CNN with pilots
    cnn = ConvolutionalNNPilot(n_lags=45, n_hidden_units=5, n_hidden_layers=5, learning_rate=1e-3, samples_per_symbol=samples_per_symbol_out,
                               batch_size=1000)
    cnn.initialize_optimizer()
    __ = cnn.fit(rx, syms)
    y_eq_cnn = cnn.apply(rx_val)

    # Loop, calculate SER and plot constellation.
    eqdict = {
        "Noisy symbols": rx[::samples_per_symbol_out],
        "VAE": y_eq,
        "Linear FFE": y_eq_lms,
        "CNN": y_eq_cnn
    }
    
    ncols = 2
    fig, ax = plt.subplots(ncols=ncols, nrows=int(np.ceil(len(eqdict)/ncols)))
    ax = ax.flatten()

    for e, (elabel, esig) in enumerate(eqdict.items()):
        # Make "constellation plot"
        ax[e].hist(esig, bins=100)

        # Calculate error metrics - Symbol Error Rate (SER)
        this_ser, __ = calc_ser_pam(esig, syms_val)

        ax[e].set_title(f"{elabel}: {this_ser:.3e}")

    plt.show()
