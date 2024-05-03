
# Module that encompasses data generation for these digital communication systems
from matplotlib.pylab import Generator
import numpy as np
import komm
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


class WienerHammersteinSystem(object):
    """
        Wiener-Hammerstein model consists of two FIR filters with a non-linearity sandwiched in-between.
        We use a third order polynomial as the non-linearity.
    """
    def __init__(self, fir1, fir2, poly_coefs, sps) -> None:
        assert len(poly_coefs) == 3
        self.fir1 = np.zeros((sps * (len(fir1) - 1) + 1, ))  # usample FIR coefficients
        self.fir1[::sps] = fir1
        self.fir2 = np.zeros((sps * (len(fir2) - 1) + 1, ))
        self.fir2[::sps] = fir2
        self.poly_coefs = np.zeros((4,))
        self.poly_coefs[0:3] = poly_coefs[::-1]  # prep for np.polyval
        self.linear_response_norm = np.linalg.norm(np.convolve(self.fir1, self.fir2))

    def forward(self, x):
        z = np.convolve(x, self.fir1, mode='same')
        z = np.polyval(self.poly_coefs, z)
        return np.convolve(z, self.fir2, mode='same') / self.linear_response_norm

class NonLinearISI(TransmissionSystem):
    """
        General non-linear channel based on a Wiener-Hammerstein (WH) model

        Blockdiagram:
            syms -> up-sampling -> RRC -> channel (WH) -> RRC -> downsampling
    """
    def __init__(self, oversampling, wh_config: dict,
                       snr_db, samples_pr_symbol: int, constellation, random_obj: Generator,
                       rrc_length=255, rrc_rolloff=0.1) -> None:
        super().__init__(samples_pr_symbol, constellation, random_obj)

        self.oversampling = oversampling
        self.snr_db = snr_db

        # Generate RRC filter
        self.pulse = rrcosfilter(rrc_length + 1, rrc_rolloff, 1.0, 1.0 / oversampling)[1]
        self.pulse = self.pulse[1::]
        self.pulse /= np.linalg.norm(self.pulse)  # normalize such the pulse has unit norm

        # Initialize Wiener-Hammerstein system
        self.wh = WienerHammersteinSystem(sps=self.oversampling, **wh_config)

    def generate_data(self, n_symbols):
        # Generate random symbols
        n_extra_symbols = len(self.wh.fir1) + len(self.wh.fir2)
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
