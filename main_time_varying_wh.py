"""
    Script that shows tracking ability of the different frameworks

    Plot loss function convergence in time-warying WH channel

"""

import komm
import matplotlib.pyplot as plt
import numpy as np
import torch

from lib.data_generation import TimeVaryingNonLinearISI
from lib.real_equalization import TorchLMSPilot, SecondVolterraPilot, SecondVolterraVAE, SecondVolterraVOLVO
from lib.utility import calc_ser_pam, calc_ser_from_probs


if __name__ == "__main__":
    # Parameters to be used
    samples_per_symbol_in = 4
    samples_per_symbol_out = 2
    seed = 124545
    snr_db = 16.0
    N_symbols = int(10e6)
    change_point = int(1e6)  # in symbols
    N_symbols_val = int(1e6)  # number of symbols used for SER calculation
    batch_size = 500  # in symbols

    n_data_chunks = N_symbols // change_point

    n_eq_taps = 25
    n_eq2_taps = 15
    channel_memory = 15  # for VAEs

    # Channel config
    wh_config = {
        "fir1": [1.0, 0.4, -0.45],  # FIR filter with two zeros (at 0.5, -0.9)
        "fir1_alt": [1., 0.5, 0.1525],  # FIR filter with two zeros (at -0.25 \pm 0.3j)
        "fir2": [ 1., -0.2, 0.02],  # FIR filter with two zeros (at 0.1 \pm 0.1j)
        "nl_type": 'poly',
        "poly_coefs": [0.9, 0.1, 0.0]
    }

    # Create modulation scheme
    order = 4
    modulation_scheme = komm.PAModulation(order=order)
    symbol_energy = np.average(np.square(np.absolute(modulation_scheme.constellation)))
    constellation = modulation_scheme.constellation
    print(f"Constellation of order {order} is: {constellation}")
    print(f"Avg. symbol energy: {symbol_energy}")

    # Generate data - training and validation
    random_obj = np.random.default_rng(seed)
    tv_whsys = TimeVaryingNonLinearISI(oversampling=samples_per_symbol_in,
                                       wh_config=wh_config,
                                       snr_db=snr_db,
                                       samples_pr_symbol=samples_per_symbol_out,
                                       constellation=constellation,
                                       random_obj=random_obj,
                                       rrc_length=samples_per_symbol_in * 8 - 1,
                                       rrc_rolloff=0.1)


    # Allocate equalisers
    # Create VAE object and process samples
    vol2_volvo = SecondVolterraVOLVO(
        channel_memory=channel_memory,
        equaliser_n_lags1=n_eq_taps,
        equaliser_n_lags2=n_eq2_taps,
        learning_rate=5e-4,
        constellation=constellation,
        batch_size=batch_size,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cuda:0"),
        dtype=torch.float32,
        lr_schedule=None
    )
    vol2_volvo.initialize_optimizer()

    # VAE
    vol2_vae = SecondVolterraVAE(
        channel_memory=channel_memory,
        equaliser_n_lags1=n_eq_taps,
        equaliser_n_lags2=n_eq2_taps,
        learning_rate=1e-3,
        constellation=constellation,
        batch_size=batch_size,
        samples_per_symbol=samples_per_symbol_out,
        noise_variance=10 ** (-snr_db / 10),
        adaptive_noise_variance=True,
        torch_device=torch.device("cpu"),
        dtype=torch.float32,
        lr_schedule=None
    )
    vol2_vae.initialize_optimizer()
    
    # Volterra with pilots
    volterra_pilot = SecondVolterraPilot(n_lags1=n_eq_taps, n_lags2=n_eq2_taps, learning_rate=5e-3,
                                samples_per_symbol=samples_per_symbol_out, batch_size=batch_size,
                                dtype=torch.float32, 
                                torch_device=torch.device("cpu"),
                                lr_schedule=None)
    volterra_pilot.initialize_optimizer()

    # LMS with pilots
    lms_pilot = TorchLMSPilot(n_taps=n_eq_taps, learning_rate=1e-3, samples_per_symbol=samples_per_symbol_out,
                              batch_size=batch_size, lr_schedule=None)
    lms_pilot.initialize_optimizer()


    # Run a change-point experiment pr. equalizer
    eqs = [vol2_volvo, vol2_vae, volterra_pilot, lms_pilot]
    res_dicts = []

    for equalizer in eqs:
        # Loop over data chunks
        loss_curve = []
        ser_pr_chunk = []
        for nchunk in range(n_data_chunks):
            # Generate a chunk of data
            rx, a = tv_whsys.generate_data(change_point)

            # Fit equaliser to this part
            mse_loss = True
            if "Pilot" in f"{equalizer}":
                __, loss = equalizer.fit(rx, a)
            else:
                mse_loss = False
                __, loss = equalizer.fit(rx)

            # Append to loss_curve
            loss_curve.append(loss)

            # Calculate SER on a test set
            rx_val, a_val = tv_whsys.generate_data(N_symbols_val)
            if equalizer.IS_PROBABILISTIC:
                q_val = equalizer.estimate_symbol_probs(rx_val)
                ser, __ = calc_ser_from_probs(q_val, a_val, equalizer.constellation.cpu().detach().numpy())
            else:
                y_hat = equalizer.apply(rx_val)
                ser, __ = calc_ser_pam(y_hat, a_val)

            ser_pr_chunk.append(ser)

            # Finally induce change point
            print(f"WH system changed!")
            tv_whsys.change_wh()

        this_res_dict = dict()
        this_res_dict['method'] = f"{equalizer}"
        this_res_dict['loss'] = np.concatenate(loss_curve)
        this_res_dict['loss_type'] = "MSE" if mse_loss else "ELBO"
        this_res_dict['ser_pr_chunk'] = ser_pr_chunk

        res_dicts.append(this_res_dict)
        

    fig, axs = plt.subplots(figsize=(10, 5), ncols=2)
    for res in res_dicts:
        if res['loss_type'] == "MSE":
            axs[0].plot(res['loss'], label=res['method'])
        else:
            axs[1].plot(res['loss'], label=res['method'])

        print(f"{res['method']}. Average SER - Type 1: {np.average(res['ser_pr_chunk'][0::2]):.3e}, Type 2: {np.average(res['ser_pr_chunk'][1::2]):.3e}")

    for ax in axs:
        ax.legend()
        ax.set_xlabel('Batches')
        ax.grid()

    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('(V2)VAE Loss')

    plt.show()