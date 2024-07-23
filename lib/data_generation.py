
# Module that encompasses data generation for these digital communication systems
from matplotlib.pylab import Generator
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import bessel, lfilter, butter, firwin
from commpy.filters import rrcosfilter

from .utility import symbol_sync, find_max_variance_sample


class TransmissionSystem(object):
    """
        Base class for all data generation.
    """
    def __init__(self, samples_pr_symbol: int, constellation, random_obj: np.random.Generator) -> None:
        self.sps = samples_pr_symbol
        self.constellation = constellation
        self.constellation_order = len(constellation)
        self.constellation_energy = np.average(np.square(np.abs(constellation)))
        self.random_obj = random_obj
        self.EsN0_db = None

    def _generate_symbols(self, n_symbols):
        return self.random_obj.choice(self.constellation, size=(n_symbols,), replace=True)

    # weak implementation
    def generate_data(self, n_symbols):
        raise NotImplementedError


class SymbolAWGNwithISI(TransmissionSystem):
    """
        Additive white Gaussian noise channel with sps = 1
        Intersymbol-interference (ISI) modeled by an FIR filter (h_isi)
    """
    def __init__(self, h_isi, snr_db, constellation, random_obj: np.random.Generator) -> None:
        super().__init__(samples_pr_symbol=1, constellation=constellation, random_obj=random_obj)
        self.snr_db = snr_db
        self.h_isi = h_isi / np.linalg.norm(h_isi)  # assert than ISI has unit norm

    def generate_data(self, n_symbols):
        n_symbols_conv = n_symbols + len(self.h_isi)  # extra symbols to pad with for ISI response
        a = self._generate_symbols(n_symbols_conv)

        # Construct Tx signal (simple 1 symbol pr sample) and add noise
        x = np.convolve(a, self.h_isi, mode='valid')
        noise_std = np.sqrt(self.constellation_energy / (2 * 10**(self.snr_db / 10)))
        awgn = noise_std * self.random_obj.standard_normal(len(x))
        if np.iscomplexobj(x):
            awgn = awgn.astype(np.complex128)
            awgn += 1j * noise_std * self.random_obj.standard_normal(len(x))

        rx = x + awgn
        rx = rx[0:n_symbols]  # truncate to valid symbols
        a = a[(len(self.h_isi) - 1):(n_symbols + len(self.h_isi) - 1)]

        # Calculate resulting EsN0
        self.EsN0_db = 10.0 * np.log10(1.0 / noise_std ** 2)

        return rx, a


class SymbolNonLinearISI(TransmissionSystem):
    """
        Symbol-level channel (1 sps) with a non-linear channel
        Intersymbol-interference (ISI) modeled by a Wiener-Hammerstein system
    """
    def __init__(self, wh_config: dict, snr_db, constellation, random_obj: np.random.Generator) -> None:
        super().__init__(samples_pr_symbol=1, constellation=constellation, random_obj=random_obj)
        self.snr_db = snr_db

        # Initailize Wiener-Hammerstein object
        self.wh = WienerHammersteinSystem(sps=1, **wh_config)
        self.wh_length = len(self.wh.fir1) + len(self.wh.fir2)

    def generate_data(self, n_symbols):
        n_symbols_conv = n_symbols + self.wh_length  # extra symbols to pad with for ISI response
        a = self._generate_symbols(n_symbols_conv)

        # Construct Tx signal (simple 1 symbol pr sample) and add noise
        x = self.wh.forward(a)
        Es = np.average(np.square(np.absolute(x)))
        noise_std = np.sqrt(Es / (2 * 10**(self.snr_db / 10)))
        awgn = noise_std * self.random_obj.standard_normal(len(x))
        if np.iscomplexobj(x):
            awgn = awgn.astype(np.complex128)
            awgn += 1j * noise_std * self.random_obj.standard_normal(len(x))
        rx = x + awgn

        # Synchronize symbols to Rx
        a = symbol_sync(rx, a, self.sps)[0:n_symbols]
        rx = rx[0:n_symbols*self.sps]

        # Calculate resulting EsN0
        base_power_pam = Es * (3 / (self.constellation_order**2 - 1))
        self.EsN0_db = 10.0 * np.log10(base_power_pam / noise_std ** 2)

        return rx, a


class PulseShapedAWGN(TransmissionSystem):
    """
        Additive white Gaussian noise channel with RRC filtered data
        Channel ISI is expected to be at symbol level and will be upsampled by the same
        upsampling factor as the rest of the system (follows practice from Lauinger et al. 2022)

        Blockdiagram:
            syms -> up-sampling -> RRC -> channel (linear ISI) -> RRC -> downsampling
    """
    def __init__(self, oversampling, h_isi,
                       snr_db, samples_pr_symbol: int, constellation, random_obj: Generator,
                       rrc_length=255, rrc_rolloff=0.1) -> None:
        super().__init__(samples_pr_symbol=samples_pr_symbol,
                         constellation=constellation,
                         random_obj=random_obj)

        self.oversampling = oversampling
        self.h_isi = h_isi
        self.snr_db = snr_db

        # Generate RRC filter
        self.pulse = rrcosfilter(rrc_length + 1, rrc_rolloff, 1.0, 1.0 / oversampling)[1]
        self.pulse = self.pulse[1::]
        self.pulse /= np.linalg.norm(self.pulse)  # normalize such the pulse has unit norm


    def generate_data(self, n_symbols):
        # Generate symbol sequence
        a = self._generate_symbols(n_symbols)

        # Upsample symbol sequence and pulse shape
        a_zeropadded = np.zeros(self.oversampling * len(a), dtype=a.dtype)
        a_zeropadded[0::self.oversampling] = a
        x = np.convolve(a_zeropadded, self.pulse)
        gg = np.convolve(self.pulse, self.pulse[::-1])
        pulse_energy = np.max(gg)
        sync_point = np.argmax(gg)

        # Calculate empirical energy pr. symbol period
        Es_emp = np.mean(np.sum(np.square(np.reshape((x - x.mean())[0:n_symbols * self.oversampling], (-1, self.oversampling))), axis=1))

        # Derive noise std
        noise_std = np.sqrt(Es_emp / (2 * 10.0 **(self.snr_db / 10.0)))
        rx = x + noise_std * self.random_obj.standard_normal(len(x))
        if np.iscomplexobj(self.constellation):
            rx += 1j * noise_std * self.random_obj.standard_normal(len(x))

        # Apply matched filter, sample recovery and decimation
        rx = np.convolve(rx, self.pulse[::-1]) / pulse_energy
        rx = rx[sync_point:-sync_point]
        max_var_samp = find_max_variance_sample(rx, self.oversampling)
        decimation_factor = int(self.oversampling / self.sps)
        assert self.oversampling % self.sps == 0
        rx = np.roll(rx, -max_var_samp)[::decimation_factor]

        # Sync symbols
        a = symbol_sync(rx, a, self.sps)[0:n_symbols]
        rx = rx[0:n_symbols*self.sps]

        # Calculate resulting EsN0
        base_power_pam = Es_emp * (3 / (self.constellation_order**2 - 1))
        self.EsN0_db = 10.0 * np.log10(base_power_pam / noise_std ** 2)

        return rx, a


class Polynomial(object):
    def __init__(self, poly_coefs) -> None:
        assert len(poly_coefs) == 3
        self.poly_coefs = np.zeros((4,))
        self.poly_coefs[0:3] = poly_coefs[::-1]  # prep for np.polyval

    def __call__(self, x):
        return np.polyval(self.poly_coefs, x)


class SplineElectroAbsorptionModulator(object):
    """
        Implementation of and electro absorption modulator (EAM) as used in

        E. M. Liang and J. M. Kahn,
        “Geometric Shaping for Distortion-Limited Intensity Modulation/Direct Detection Data Center Links,”
        IEEE Photonics Journal, vol. 15, no. 6, pp. 1–17, 2023, doi: 10.1109/JPHOT.2023.3335398.

        NB! No chirp.
    """

    # Voltage-to-absorption curve directly from (Liang and Kahn, 2023)
    ABSORPTION_KNEE_POINTS_X = np.flip(np.array([0.0, -0.5,  -1.0, -2.0, -3.0, -3.5,  -3.8]), (0,))  # driving voltage
    ABSORPTION_KNEE_POINTS_Y = np.flip(np.array([0.0, 1.25,   2.5,  5.0,  9.5, 13.0,  12.5]), (0,))  # absorption in dB

    def __init__(self, laser_power_dbm, dac_vpp, dac_vb) -> None:
        self.Pin = 10 ** (laser_power_dbm / 10) * 1e-3  # [Watt]
        self.alpha_db = CubicSpline(self.ABSORPTION_KNEE_POINTS_X, self.ABSORPTION_KNEE_POINTS_Y)
        self.dac_vpp = dac_vpp
        self.dac_vb = dac_vb

    def __call__(self, x):
        v = (x - np.min(x)) / (np.max(x) - np.min(x))
        v = self.dac_vpp * v + self.dac_vb
        absorp = self.alpha_db(v)
        return self.Pin * np.float_power(10.0, -absorp / 10.0)  # ideal detection, removes sqrt
    

class WienerHammersteinSystem(object):
    """
        Wiener-Hammerstein model consists of two FIR filters with a non-linearity sandwiched in-between.
        We use a third order polynomial as the non-linearity by default.
    """
    def __init__(self, fir1, fir2, sps, nl_type='poly', **nl_config) -> None:
        self.fir1 = np.zeros((sps * (len(fir1) - 1) + 1, ))  # usample FIR coefficients
        self.fir1[::sps] = fir1
        self.fir1 /= np.linalg.norm(self.fir1)  # ensure unit norm
        self.fir2 = np.zeros((sps * (len(fir2) - 1) + 1, ))
        self.fir2[::sps] = fir2
        self.fir2 /= np.linalg.norm(self.fir2)  # ensure unit norm

        # Determine what non-linearity that should be used in the middle
        self.nl = lambda x: x
        if nl_type == 'poly':
            self.nl = Polynomial(**nl_config)
        elif nl_type == 'eam':
            self.nl = SplineElectroAbsorptionModulator(**nl_config)
        else:
            raise Exception(f"Unknown nonlinearity in WH: '{nl_type}'")

        # FIXME: Make correct delay estimate
        self.delay = len(self.fir1) + len(self.fir2)

    def forward(self, x):
        z = np.convolve(x, self.fir1, mode='same')
        z = self.nl(z)
        z = np.convolve(z, self.fir2, mode='same')
        return (z - np.mean(z)) / np.std(z)


class LowPassFilter(object):
    def __init__(self, lp_type, order, cutoff) -> None:
        self.lp_type = lp_type
        self.order = order
        self.cutoff = cutoff

        self.lp_filter_b, self.lp_filter_a = 1, 1
        
        if lp_type == 'bessel':
            self.lp_filter_b, self.lp_filter_a = bessel(order, cutoff, norm='mag')
        elif lp_type == 'butter':
            self.lp_filter_b, self.lp_filter_a = butter(order, cutoff)
        elif lp_type == 'firwin':
            self.lp_filter_b, self.lp_filter_a = firwin(order, cutoff), 1
        else:
            raise Exception(f"Unknown low pass filter type: '{lp_type}'")

    def apply(self, x):
        return lfilter(self.lp_filter_b, self.lp_filter_a, x)
    
    def __repr__(self) -> str:
        return f"{self.lp_type}({self.order}, Wn={self.cutoff})"


class LowPassWienerHammersteinSystem(object):
    """
        Wiener-Hammerstein model but with low-pass filter
        lowpass filter can be either: bessel, butter or firwin
        We use a third order polynomial as the non-linearity by default.
    """
    def __init__(self, lp1_config, lp2_config, nl_type='poly', **nl_config) -> None:
        # Bessel filter initialization
        self.filter1 = LowPassFilter(**lp1_config)
        self.filter2 = LowPassFilter(**lp2_config)

        # Determine what non-linearity that should be used in the middle
        self.nl = lambda x: x
        if nl_type == 'poly':
            self.nl = Polynomial(**nl_config)
        elif nl_type == 'eam':
            self.nl = SplineElectroAbsorptionModulator(**nl_config)
        else:
            raise Exception(f"Unknown nonlinearity in WH: '{nl_type}'")

        # FIXME: Make correct delay estimate
        self.delay = self.filter1.order + self.filter2.order

    def forward(self, x):
        z = self.filter1.apply(x)
        z = self.nl(z)
        z = self.filter2.apply(z)
        return (z - np.mean(z)) / np.std(z)


class NonLinearISI(TransmissionSystem):
    """
        General non-linear channel based on a Wiener-Hammerstein (WH) model

        Blockdiagram:
            syms -> up-sampling -> RRC -> channel (WH) -> RRC -> downsampling
    """
    def __init__(self, oversampling, wh_config: dict,
                       snr_db, samples_pr_symbol: int, constellation, random_obj: Generator,
                       rrc_length=255, rrc_rolloff=0.1, wh_type='fir') -> None:
        super().__init__(samples_pr_symbol, constellation, random_obj)

        self.oversampling = oversampling
        self.snr_db = snr_db

        # Generate RRC filter
        self.pulse = rrcosfilter(rrc_length + 1, rrc_rolloff, 1.0, 1.0 / oversampling)[1]
        self.pulse = self.pulse[1::]
        self.pulse /= np.linalg.norm(self.pulse)  # normalize such the pulse has unit norm

        # Initialize Wiener-Hammerstein system
        self.wh = lambda x: x
        if wh_type == 'fir':
            self.wh = WienerHammersteinSystem(sps=self.oversampling, **wh_config)
        elif wh_type == 'lp':
            self.wh = LowPassWienerHammersteinSystem(**wh_config)
        else:
            raise Exception(f"Unknown WH type: '{wh_type}'")

    def generate_data(self, n_symbols):
        # Generate random symbols
        n_extra_symbols = self.wh.delay
        a = self._generate_symbols(n_symbols + n_extra_symbols)

        # Upsample symbol sequence and pulse shape
        a_zeropadded = np.zeros(self.oversampling * (n_symbols + n_extra_symbols), dtype=a.dtype)
        a_zeropadded[0::self.oversampling] = a
        x = np.convolve(a_zeropadded, self.pulse)
        gg = np.convolve(self.pulse, self.pulse[::-1])
        pulse_energy = np.max(gg)
        sync_point = np.argmax(gg)

        # AWGN channel with Wiener-Hammerstein non-linearity
        rx = self.wh.forward(x)

        # Calculate empirical energy pr. symbol period
        Es_emp = np.mean(np.sum(np.square(np.reshape((rx - rx.mean())[0:n_symbols * self.oversampling], (-1, self.oversampling))), axis=1))

        # Derive noise std
        noise_std = np.sqrt(Es_emp / (2 * 10.0 **(self.snr_db / 10.0)))
        rx += noise_std * self.random_obj.standard_normal(len(rx))
        if np.iscomplexobj(self.constellation):
            rx += 1j * noise_std * self.random_obj.standard_normal(len(rx))

        # Matched filter, sample recovery and decimation
        rx = np.convolve(rx, self.pulse[::-1]) / pulse_energy
        rx = rx[sync_point:-sync_point]
        max_var_samp = find_max_variance_sample(rx, self.oversampling)
        decimation_factor = int(self.oversampling / self.sps)
        assert self.oversampling % self.sps == 0
        rx = np.roll(rx, -max_var_samp)[::decimation_factor]

        # Sync symbols
        a = symbol_sync(rx, a, self.sps)[0:n_symbols]
        rx = rx[0:n_symbols*self.sps]

        # Calulate pr. symbol SNR
        base_power_pam = Es_emp * (3 / (self.constellation_order**2 - 1))
        self.EsN0_db = 10.0 * np.log10(base_power_pam / noise_std ** 2)

        return rx, a


class LightEmittingDiode(object):
    """
        LED ODE model from

        [1] X. Deng et al., “Mitigating LED Nonlinearity to Enhance Visible Light Communications,” IEEE Transactions on Communications, vol. 66, no. 11, pp. 5593–5607, Nov. 2018, doi: 10.1109/TCOMM.2018.2858239.

        Solved by applying Eulers method
    """
    def __init__(self, optical_wavelength, active_layer_thickness,
                 active_layer_area, Anr, Br, Cnr, doping_concentration) -> None:
        # Populate important attributes
        self.optical_wavelength = optical_wavelength  # assumed to be in [nm]
        optical_frequency = 3e8 / (self.optical_wavelength * 1e-9)

        # Physical constants
        self.electron_charge = 1.6e-19
        self.planck_constant = 6.626e-34  # [J/Hz]

        # Derived constants following notation in [1]
        self.energy_pr_photon = self.planck_constant * optical_frequency  # [J]
        self.a0 = 1 / (self.electron_charge * active_layer_thickness * active_layer_area)
        self.a1 = Br * doping_concentration + Anr
        self.a2 = Br
        self.a3 = Cnr
        self.a4 = self.energy_pr_photon * active_layer_area * active_layer_thickness * Br * doping_concentration
        self.a5 = self.energy_pr_photon * active_layer_area * active_layer_thickness * Br
    
    def forward(self, current, fs):
        # Find time interval to solve the ode in
        ts = 1/fs

        # Apply Eulers method, i.e.  n_c(t+1) \approx n_c(t) + Ts * dn_c(t) / dt
        nc_sol = np.zeros_like(current)
        current_copy = np.copy(current)
        nc_sol[0] = current_copy[0]  # initial condition
        for i in range(1, len(current)):
            nc_sol[i] = ts * current_copy[i] * self.a0 + (1 - self.a1 * ts) * nc_sol[i-1]\
                - self.a2 * ts * nc_sol[i-1]**2 - self.a3 * ts * nc_sol[i-1]**3 
        
        # Compute optical power
        return self.a4 * nc_sol + self.a5 * nc_sol**2


class LightEmittingDiodeSystem(TransmissionSystem):
    """
        LED free-space simulation model
        LED response is calculated by numerical integration of the rate equations

        Blockdiagram:
            syms -> up-sampling -> RRC -> LED -> Photodiode (square + AWGN) -> RRC -> downsampling/alignment
    """
    def __init__(self, baud_rate, oversampling, led_config: dict, current_pp: float,
                 snr_db, samples_pr_symbol: int, constellation, random_obj: Generator,
                 rrc_length=255, rrc_rolloff=0.1, current_bias: float=0.0) -> None:
        super().__init__(samples_pr_symbol, constellation, random_obj)

        self.baud_rate = baud_rate
        self.oversampling = oversampling
        self.snr_db = snr_db
        self.fs = baud_rate * oversampling
        self.current_pp = current_pp
        self.current_bias = current_bias

        # Generate RRC filter
        self.pulse = rrcosfilter(rrc_length + 1, rrc_rolloff, 1 / baud_rate, self.fs)[1]
        self.pulse = self.pulse[1::]
        self.pulse /= np.linalg.norm(self.pulse)  # normalize such the pulse has unit norm

        # Initialize LED
        self.led = LightEmittingDiode(**led_config)
        

    def generate_data(self, n_symbols):
        # Generate random symbols
        n_extra_symbols = 0
        a = self._generate_symbols(n_symbols + n_extra_symbols)

        # Upsample symbol sequence and pulse shape
        a_zeropadded = np.zeros(self.oversampling * (n_symbols + n_extra_symbols), dtype=a.dtype)
        a_zeropadded[0::self.oversampling] = a
        x = np.convolve(a_zeropadded, self.pulse)
        gg = np.convolve(self.pulse, self.pulse[::-1])
        pulse_energy = np.max(gg)
        sync_point = np.argmax(gg)

        # Conver to a current - normalize digital signal to [-1, 1] and apply transform (1 + x) * current_dc
        x = (x / np.max(np.abs(x)) + 1) * self.current_pp + self.current_bias

        # Apply LED model
        yled = self.led.forward(x, self.fs)

        # Photoiode (square-law + AWGN)
        rx = np.square(yled)

        # Calculate empirical energy pr. symbol period
        Es_emp = np.mean(np.sum(np.square(np.reshape((rx - rx.mean())[0:n_symbols * self.oversampling], (-1, self.oversampling))), axis=1))

        # Derive noise std
        noise_std = np.sqrt(Es_emp / (2 * 10.0 **(self.snr_db / 10.0)))
        rx += noise_std * self.random_obj.standard_normal(len(rx))
        if np.iscomplexobj(self.constellation):
            rx += 1j * noise_std * self.random_obj.standard_normal(len(rx))

        # Normalize
        rx = (rx - np.mean(rx)) / np.std(rx)

        # Matched filter, sample recovery and decimation
        rx = np.convolve(rx, self.pulse[::-1]) / pulse_energy
        rx = rx[sync_point:-sync_point]
        max_var_samp = find_max_variance_sample(rx, self.oversampling)
        decimation_factor = int(self.oversampling / self.sps)
        assert self.oversampling % self.sps == 0
        rx = np.roll(rx, -max_var_samp)[::decimation_factor]

        # Sync symbols
        a = symbol_sync(rx, a, self.sps)[0:n_symbols]
        rx = rx[0:n_symbols*self.sps]

        # Calulate pr. symbol SNR
        base_power_pam = Es_emp * (3 / (self.constellation_order**2 - 1))
        self.EsN0_db = 10.0 * np.log10(base_power_pam / noise_std ** 2)

        return rx, a