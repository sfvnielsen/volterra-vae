from itertools import combinations_with_replacement

import numpy as np
import numpy.typing as npt
import torch
import torchaudio.functional as taf
from scipy.special import comb


class Passthrough(object):
    def __init__(self, samples_per_symbol) -> None:
        self.samples_per_symbol = samples_per_symbol

    def fit(self, x):
        return x[::self.samples_per_symbol]

    def apply(self, x):
        return x[::self.samples_per_symbol]

    def __repr__(self) -> str:
        return "No equaliser"


class Equalizer(object):
    def __init__(self, n_taps, learning_rate, samples_per_symbol=1, reference_tap=None) -> None:
        self.n_taps = n_taps
        self.learning_rate = learning_rate
        self.reference_tap = reference_tap
        if self.reference_tap is None:
            self.reference_tap = n_taps // 2
        self.n_pad_start = self.n_taps - self.reference_tap - 1  # zeropadding when ref tap central = int(np.floor(self.n_taps / 2))
        self.n_pad_end = self.n_taps - self.n_pad_start
        self.samples_per_symbol = samples_per_symbol

        assert (n_taps >= samples_per_symbol)

        # Initialize filter
        self.filter = np.zeros((n_taps,), dtype='float64')
        self.filter[self.reference_tap] = 1.0

    def _check_input(self, input_array):
        n_symbols = len(input_array) // self.samples_per_symbol
        if n_symbols * self.samples_per_symbol != len(input_array):
            print(f"Warning! Number of samples in receiver signal ({len(input_array)}) does not match with a multiplum of the symbol length ({self.samples_per_symbol})")
        return n_symbols

    def get_filter(self):
        return self.filter

    def set_filter(self, new_filter):
        assert (len(new_filter) == self.n_taps)
        self.filter = new_filter

    # weak implementation
    def apply(self, receiver_signal):
        raise NotImplementedError

    # weak implementation
    def __repr__(self) -> str:
        raise NotImplementedError


class LMSPilot(Equalizer):
    def __init__(self, n_taps, learning_rate, samples_per_symbol=1, reference_tap=None) -> None:
        super().__init__(n_taps, learning_rate, samples_per_symbol, reference_tap)
        self.phase_shift = 0.0

    def __repr__(self) -> str:
        return "LMSPilot(Real)"

    def fit(self, receiver_signal, tx_symbol):
        # Check input and figure out how many symbols that are being transmitted
        n_symbols = self._check_input(receiver_signal)

        if n_symbols != len(tx_symbol):
            raise Exception(f"Number of receiver samples does not match with the number of supplied tx symbols!")

        # Allocate output arrays
        eq_out = np.zeros((n_symbols, ), dtype=receiver_signal.dtype)

        # Zero pad input signal to account for delay to reference tap
        receiver_signal_padded = np.concatenate((np.zeros((self.n_pad_start,), dtype=receiver_signal.dtype),
                                                 receiver_signal,
                                                 np.zeros((self.n_pad_end,), dtype=receiver_signal.dtype)))

        # Loop over input signals
        for i in range(0, n_symbols):
            # Get signal slice
            sigslice = slice(i * self.samples_per_symbol, i * self.samples_per_symbol + self.n_taps)

            # Equalizer output and error signal
            delayline = np.flipud(receiver_signal_padded[sigslice])
            eq_out[i] = np.sum(delayline * self.filter)
            error = eq_out[i] - tx_symbol[i]

            # Update filter
            self.filter = self.filter - self.learning_rate * delayline * error

            # If filter has exploded, terminate
            if np.any(np.isnan(self.filter)):
                print('WARNING! Filter contains NaN. Terminating fit call.')
                break

        return eq_out

    def apply(self, receiver_signal):
        # Check input and figure out how many symbols that are being transmitted
        n_symbols = self._check_input(receiver_signal)
        eq_out = np.zeros((n_symbols, ), dtype=receiver_signal.dtype)

        receiver_signal_padded = np.concatenate((np.zeros((self.n_pad_start,), dtype=receiver_signal.dtype),
                                                 receiver_signal,
                                                 np.zeros((self.n_pad_end,), dtype=receiver_signal.dtype)))

        # Loop over input signals
        for i in range(n_symbols):
            # Get signal slice
            sigslice = slice(i * self.samples_per_symbol, i * self.samples_per_symbol + self.n_taps)

            # Equalizer output and error signal
            eq_out[i] = np.sum(np.flipud(receiver_signal_padded[sigslice]) * self.filter)

        return eq_out


class GenericTorchBlindEqualizer(object):
    """
        Parent class for blind-equalizers with torch optimization
    """
    def __init__(self, samples_per_symbol, batch_size, dtype=torch.float32, torch_device=torch.device("cpu"), flex_update_interval=None) -> None:
        # FIXME: "Flex" update scheme from (Lauinger, 2022)
        self.samples_per_symbol = samples_per_symbol
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.dtype = dtype
        self.loss_print_interval = 100  # FIXME: Make part of constructor

    def _check_input(self, input_array):
        n_batches = len(input_array) // self.samples_per_symbol // self.batch_size
        if n_batches * self.samples_per_symbol * self.batch_size != len(input_array):
            print(f"Warning! Number of samples in receiver signal does not match with a multiplum of the symbol length ({self.samples_per_symbol})")
        return n_batches


    def fit(self, y_input):
        # Check input
        n_batches = self._check_input(y_input)

        # Copy input array to device
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)

        # Allocate output arrays - as torch tensors
        y_eq = torch.zeros((y.shape[-1] // self.samples_per_symbol, ), dtype=y.dtype).to(device=self.torch_device)

        # Loop over batches
        for n in range(n_batches):
            this_slice = slice(n * self.batch_size * self.samples_per_symbol,
                                n * self.batch_size * self.samples_per_symbol + self.batch_size * self.samples_per_symbol)

            xhat = self.forward(y[this_slice])

            loss = self._calculate_loss(xhat, y[this_slice])

            if n % self.loss_print_interval == 0:
                print(f"Batch {n}, Loss: {loss.item():.3f}")

            self._update_model(loss)

            y_eq[n * self.batch_size: n * self.batch_size + self.batch_size] = xhat.clone().detach()

        return y_eq.detach().cpu().numpy()

    def apply(self, y_input):
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)
        with torch.set_grad_enabled(False):
            y_eq = self.forward(y)
        return y_eq.cpu().numpy()

    # weak implementation
    def _update_model(self, loss):
        raise NotImplementedError

    # weak implementation
    def forward(self, y):
        raise NotImplementedError

    # weak implementation
    def _calculate_loss(self, xhat, y):
        raise NotImplementedError

    # weak implementation
    def print_model_parameters(self):
        raise NotImplementedError

    # weak implementation
    def train_mode(self):
        raise NotImplementedError

    # weak implementation
    def eval_mode(self):
        raise NotImplementedError

    # weak implementation
    def __repr__(self) -> str:
        raise NotImplementedError


# Linear VAE
class VAELinearForward(GenericTorchBlindEqualizer):
    """
        Parent class for all the blind-equalizer VAEs with linear channel model
    """
    def __init__(self, channel_n_taps, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), flex_update_interval=None, **equaliser_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, dtype, torch_device, flex_update_interval)

        # Channel model - in this type of VAE always a FIRfilter
        assert (samples_per_symbol <= channel_n_taps)
        self.channel_n_taps = channel_n_taps
        self.channel_filter = torch.zeros((self.channel_n_taps,), dtype=self.dtype, requires_grad=True)
        self.channel_filter.data[self.channel_n_taps // 2] = 1.0
        self.channel_filter.to(self.torch_device)

        # Equaliser model - method initialize_decoder to be implemented by each child of this class
        self.equaliser = self.initialize_equaliser(**equaliser_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Learning parameters
        self.learning_rate = learning_rate

        # Process constellation
        self.constellation = torch.from_numpy(constellation).type(self.dtype).to(self.torch_device)
        self.constellation_size = len(self.constellation)
        #self.constellation_scale = torch.sqrt(torch.mean(self.constellation**2))
        self.constellation_amp_mean = torch.mean(torch.abs(self.constellation))

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / self.constellation_size

        # Define optimizer object
        self.optimizer = torch.optim.Adam([{'params': self.channel_filter},
                                           {'params': self.equaliser.parameters()}],
                                          lr=self.learning_rate, amsgrad=True)

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def initialize_equaliser(self, **decoder_kwargs) -> torch.nn.Module:
        # weak implementation
        raise NotImplementedError

    def forward(self, y) -> torch.TensorType:
        # weak implementation
        raise NotImplementedError

    def _soft_demapping(self, xhat):
        # Produce softmax outputs from equalised signal
        # xhat is a [N] tensor (vector of inputs)
        # output is [M, N] tensor, where M is the number of unique amplitude levels for the constellation
        # FIXME: Implement the real soft-demapping from (Lauinger, 2022) based on Maxwell-Boltzmann distribution
        # NB! Lauinger has a normalization step of xhat before this. Removed because it is not needed?
        qest = torch.transpose(self.sm.forward(-(torch.outer(xhat, torch.ones_like(self.constellation)) - self.constellation)**2 / (self.noise_variance)), 1, 0)
        return qest

    def _calculate_loss(self, xhat, y):
        pre_delay = (self.channel_n_taps - 1) // 2
        end_delay = -(self.channel_n_taps - 1) // 2

        # Do soft-demapping on equalised signal
        # FIXME: How to handle models that output q-est directly?
        qest = self._soft_demapping(xhat)

        # KL Divergence term
        kl_div = torch.sum(qest[:, pre_delay:end_delay] * (torch.log(qest[:, pre_delay:end_delay] / self.constellation_prior[:, None] + self.epsilon)), dim=0)

        # Expectation of likelihood term - calculate subterms first - follow naming convention from Lauinger 2022
        expect_x = torch.zeros_like(y)  # Insert zero on elements not in multiplum of SpS
        expect_x_sq = torch.zeros_like(y)

        qc = qest * self.constellation[:, None]
        expect_x[::self.samples_per_symbol] = torch.sum(qc, dim=0)

        qc2 = qest * self.constellation[:, None] ** 2
        expect_x_sq[::self.samples_per_symbol] = torch.sum(qc2, dim=0)

        # More subterms - apply channel model to expected x
        hex = taf.convolve(expect_x, self.channel_filter, mode='valid')

        h_squared = torch.square(self.channel_filter)
        hsqconvx = taf.convolve(expect_x_sq - torch.square(expect_x), h_squared, mode='valid')
        e_hx2 = torch.sum(torch.square(hex) + hsqconvx)

        # Calculate loss - apply indexing to y to match with convolution
        exponent_term = torch.sum(torch.square(y[pre_delay:end_delay])) - 2 * torch.matmul(y[pre_delay:end_delay], hex) + e_hx2
        loss = torch.sum(kl_div) + (y.shape[-1] - self.channel_n_taps + 1) * torch.log(exponent_term)

        if self.adaptive_noise_variance:
            with torch.no_grad():
                self.noise_variance = exponent_term / (y.shape[-1] - self.channel_n_taps + 1)

        return loss

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Zero gradients
        self.noise_variance.grad = None
        self.channel_filter.grad = None
        self.equaliser.zero_grad()

    def print_model_parameters(self):
        # Print convolutional kernels
        print(f"Equaliser: {self.equaliser.weight}")
        print(f"Channel: {self.channel_filter}")

        # Print noise variance
        print(f"Noise variance: {self.noise_variance}")

    def train_mode(self):
        self.channel_filter.requires_grad = True
        self.equaliser.train()

    def eval_mode(self):
        self.channel_filter.requires_grad = False
        self.equaliser.eval()


class LinearVAE(VAELinearForward):
    def __init__(self, channel_n_taps, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, torch_device=torch.device('cpu'), flex_update_interval=None, **equaliser_kwargs) -> None:
        super().__init__(channel_n_taps, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, torch_device, flex_update_interval, **equaliser_kwargs)

    def initialize_equaliser(self, **equaliser_kwargs):
        # Equaliser FIR filter -  n_taps (padded with zeros)
        equaliser = torch.nn.Conv1d(kernel_size=equaliser_kwargs['equaliser_n_taps'], in_channels=1, out_channels=1,
                                  stride=self.samples_per_symbol, bias=False, padding=(equaliser_kwargs['equaliser_n_taps'] - 1) // 2,
                                  dtype=self.dtype)
        torch.nn.init.dirac_(equaliser.weight)
        equaliser.to(self.torch_device)
        return equaliser

    def forward(self, y):
        y_eq = self.equaliser.forward(y[None, None, :]).squeeze()
        return y_eq

    def __repr__(self) -> str:
        return "LinVAE"
