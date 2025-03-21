import numpy as np
import numpy.typing as npt
import torch
import torchaudio.functional as taf

from .torch_components import StraightThroughArgMin, SecondOrderVolterraSeries, WienerHammersteinNN, CNN
from .utility import calculate_mmse_weights


class Passthrough(object):
    def __init__(self, samples_per_symbol) -> None:
        self.samples_per_symbol = samples_per_symbol

    def fit(self, x):
        return x[::self.samples_per_symbol]

    def apply(self, x):
        return x[::self.samples_per_symbol]

    def __repr__(self) -> str:
        return "No equaliser"


class TheoreticalMMSE(object):
    def __init__(self, samples_per_symbol, h_isi, num_eq_taps, snr, ref_tap=None, input_delay=None):
        # FIXME: Currently only works on 1 SpS type problems
        assert (samples_per_symbol == 1)
        self.ref_tap = num_eq_taps // 2 if ref_tap is None else ref_tap
        self.input_delay = np.argmax(np.abs(h_isi)) if input_delay is None else input_delay
        self.filter = calculate_mmse_weights(h_isi, num_eq_taps, snr, self.ref_tap, self.input_delay)
        self.n_taps = num_eq_taps
        self.needs_cpr = False

    def fit(self, x):
        print(f"WARNING: Calling fit on Theoretical MMSE does not update weights.")
        return np.convolve(x, self.filter)[self.ref_tap:(self.ref_tap + len(x))]

    def apply(self, x):
        return np.convolve(x, self.filter)[self.ref_tap:(self.ref_tap + len(x))]

    def __repr__(self) -> str:
        return "Theoretical MMSE"


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


class GenericTorchPilotEqualizer(object):
    """ Parent class that implements a supervised equalizer (with a pilot sequence)
    """
    IS_PROBABILISTIC = False

    def __init__(self, samples_per_symbol, batch_size, learning_rate, dtype=torch.float32, torch_device=torch.device("cpu"),
                 lr_schedule='step') -> None:
        self.samples_per_symbol = samples_per_symbol
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.dtype = dtype
        self.loss_print_interval = 100
        self.optimizer = None
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule

    def _check_input(self, input_array, syms_array):
        n_batches_input = len(input_array) // self.samples_per_symbol // self.batch_size
        n_batches_syms = len(syms_array) // self.batch_size
        if n_batches_input * self.samples_per_symbol * self.batch_size != len(input_array):
            print(f"Warning! Number of samples in receiver signal does not match with a multiplum of the symbol length ({self.samples_per_symbol})")

        if n_batches_input != n_batches_syms:
            raise Exception(f"Number of supplied inputs batches ({n_batches_input}) does not match number symbol batches ({n_batches_syms})")

        return n_batches_input

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.get_parameters(),
                                          lr=self.learning_rate, amsgrad=True)

    def fit(self, y_input, tx_symbol):
        # Check input
        n_batches = self._check_input(y_input, tx_symbol)

        # Check that optimizer has been initialized
        if self.optimizer is None:
            raise Exception("Optimizer has not been initialized yet. Please call 'initialize_optimizer' before proceeding to 'fit'.")

        # Copy input and symbol arrays to device
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)
        tx_symbol = torch.from_numpy(tx_symbol).type(self.dtype).to(device=self.torch_device)

        # Allocate output arrays - as torch tensors
        y_eq = torch.zeros((y.shape[-1] // self.samples_per_symbol), dtype=y.dtype).to(device=self.torch_device)

        # Learning rate scheduler (step lr - learning rate is changed every step)
        if self.lr_schedule:
            steps = 10
            reduction = 0.01
            gamma = np.exp(np.log(reduction) / steps)
            lr_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=n_batches//steps, gamma=gamma)
        loss_curve = np.zeros((n_batches, ))

        # Loop over batches
        for n in range(n_batches):
            this_input_slice = slice(n * self.batch_size * self.samples_per_symbol,
                                     n * self.batch_size * self.samples_per_symbol + self.batch_size * self.samples_per_symbol)
            this_sym_slice = slice(n * self.batch_size,
                                   n * self.batch_size + self.batch_size)

            xhat = self.forward(y[this_input_slice])

            loss = self._calculate_loss(xhat, y[this_input_slice], tx_symbol[this_sym_slice])

            if n % self.loss_print_interval == 0:
                lrprint = lr_schedule.get_last_lr()[-1] if self.lr_schedule else self.learning_rate
                print(f"Batch {n}, Loss: {loss.item():.3f} (LR: {lrprint:.3e})")

            self._update_model(loss)

            if self.lr_schedule:
                lr_schedule.step()

            y_eq[n * self.batch_size: n * self.batch_size + self.batch_size] = xhat.clone().detach()
            loss_curve[n] = loss.item()

        return y_eq.detach().cpu().numpy(), loss_curve

    def apply(self, y_input):
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)
        with torch.no_grad():
            y_eq = self.forward(y)
        return y_eq.detach().cpu().numpy()

    # weak implementation
    def _update_model(self, loss):
        raise NotImplementedError

    # weak implementation
    def forward(self, y):
        raise NotImplementedError

    # weak implementation
    def _calculate_loss(self, xhat, y, syms):
        raise NotImplementedError

    # weak implementation
    def get_parameters(self):
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


class TorchLMSPilot(GenericTorchPilotEqualizer):
    """ Simple feed-forward equaliser
    """
    def __init__(self, n_taps, learning_rate, samples_per_symbol,
                 batch_size, dtype=torch.float32, torch_device=torch.device("cpu"), lr_schedule='step') -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        self.equaliser = torch.nn.Conv1d(kernel_size=n_taps, in_channels=1, out_channels=1,
                                         stride=self.samples_per_symbol, bias=False, padding=(n_taps - 1) // 2,
                                         dtype=self.dtype)
        torch.nn.init.dirac_(self.equaliser.weight)
        self.equaliser.to(self.torch_device)

    def __repr__(self) -> str:
        return "LMSPilot(Torch)"

    def get_filter(self):
        return self.equaliser.weight.squeeze().detach().cpu().numpy()

    def get_parameters(self):
        return self.equaliser.parameters()

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.equaliser.zero_grad()

    def forward(self, y):
        y_eq = self.equaliser.forward(y[None, None, :]).squeeze()
        return y_eq

    def _calculate_loss(self, xhat, y, syms):
        return torch.mean(torch.square(xhat - syms))

    def print_model_parameters(self):
        print(self.equaliser.weight)

    def train_mode(self):
        self.equaliser.train()

    def eval_mode(self):
        self.equaliser.eval()


class SecondVolterraPilot(GenericTorchPilotEqualizer):
    def __init__(self, n_lags1, n_lags2, learning_rate, samples_per_symbol, batch_size, dtype=torch.float32,
                 torch_device=torch.device("cpu"), lr_schedule='step') -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Initialize second order model
        self.equaliser = SecondOrderVolterraSeries(n_lags1=n_lags1, n_lags2=n_lags2, samples_per_symbol=samples_per_symbol,
                                                   dtype=dtype, torch_device=torch_device)

    def get_parameters(self):
        return self.equaliser.parameters()

    def _update_model(self, loss):
        loss.backward()
        self.optimizer.step()
        self.equaliser.zero_grad()

    def forward(self, y):
        return self.equaliser.forward(y)

    def _calculate_loss(self, xhat, y, syms):
        return torch.mean(torch.square(xhat - syms))

    # weak implementation
    def print_model_parameters(self):
        print("1st order kernel: ")
        print(f"{self.equaliser.kernel1}")

        print("2nd order kernel: ")
        print(f"{self.equaliser.kernel2}")

    # weak implementation
    def train_mode(self):
        self.equaliser.requires_grad_(True)

    # weak implementation
    def eval_mode(self):
        self.equaliser.requires_grad_(False)

    def __repr__(self) -> str:
        return f"SecondOrderVolterraPilot({len(self.equaliser.kernel1)}, {self.equaliser.kernel2.shape[0]})"


class WienerHammersteinNNPilot(GenericTorchPilotEqualizer):
    def __init__(self, n_lags, n_hidden_unlus, unlu_depth, unlu_hidden_size, learning_rate,
                 samples_per_symbol, batch_size, dtype=torch.float32, torch_device=torch.device("cpu"),
                 lr_schedule='step') -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Initialize WienerHammerStein NN
        self.equaliser = WienerHammersteinNN(n_lags=n_lags, n_hidden_unlus=n_hidden_unlus,
                                             unlu_depth=unlu_depth, unlu_hidden_size=unlu_hidden_size,
                                             samples_per_symbol=samples_per_symbol,
                                             dtype=dtype, torch_device=torch_device)

    def get_parameters(self):
        return self.equaliser.parameters()

    def _update_model(self, loss):
        loss.backward()
        self.optimizer.step()
        self.equaliser.zero_grad()

    def forward(self, y):
        return self.equaliser.forward(y)

    def _calculate_loss(self, xhat, y, syms):
        return torch.mean(torch.square(xhat - syms))

    def print_model_parameters(self):
        pass

    def train_mode(self):
        self.equaliser.requires_grad_(True)

    def eval_mode(self):
        self.equaliser.requires_grad_(False)

    def __repr__(self) -> str:
        return f"WienerHammersteinNNPilot"


class ConvolutionalNNPilot(GenericTorchPilotEqualizer):
    """
        CNN equaliser (with FC feed-forward NN)
    """
    def __init__(self, n_lags, n_hidden_units, n_hidden_layers, learning_rate,
                 samples_per_symbol, batch_size, dtype=torch.float32, torch_device=torch.device("cpu"),
                 lr_schedule='step') -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Initialize CNN
        self.equaliser = CNN(n_lags=n_lags, n_hidden_units=n_hidden_units, n_hidden_layers=n_hidden_layers,
                             samples_per_symbol=samples_per_symbol,
                             dtype=dtype, torch_device=torch_device)

        self.discard_samples = n_lags // 2

    def get_parameters(self):
        return self.equaliser.parameters()

    def _update_model(self, loss):
        loss.backward()
        self.optimizer.step()
        self.equaliser.zero_grad()

    def forward(self, y):
        return self.equaliser.forward(y)

    def _calculate_loss(self, xhat, y, syms):
        return torch.mean(torch.square(xhat[self.discard_samples:-self.discard_samples] - syms[self.discard_samples:-self.discard_samples]))

    def print_model_parameters(self):
        pass

    def train_mode(self):
        self.equaliser.requires_grad_(True)

    def eval_mode(self):
        self.equaliser.requires_grad_(False)

    def __repr__(self) -> str:
        return f"CNNPilot"


class GenericTorchBlindEqualizer(object):
    """
        Parent class for blind-equalizers with torch optimization
    """
    IS_PROBABILISTIC = False

    def __init__(self, samples_per_symbol, batch_size, learning_rate, dtype=torch.float32, torch_device=torch.device("cpu"),
                 lr_schedule='step') -> None:
        self.samples_per_symbol = samples_per_symbol
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.dtype = dtype
        self.loss_print_interval = 100  # FIXME: Make part of constructor
        self.learning_rate = learning_rate
        self.optimizer = None
        self.lr_schedule = lr_schedule

    def _check_input(self, input_array):
        n_batches = len(input_array) // self.samples_per_symbol // self.batch_size
        if n_batches * self.samples_per_symbol * self.batch_size != len(input_array):
            print(f"Warning! Number of samples in receiver signal does not match with a multiplum of the symbol length ({self.samples_per_symbol})")
        return n_batches

    def initialize_optimizer(self):
        self.optimizer = torch.optim.Adam(self.get_parameters(),
                                          lr=self.learning_rate, amsgrad=True)

    def fit(self, y_input):
        # Check input
        n_batches = self._check_input(y_input)

        # Check that optimizer has been initialized
        if self.optimizer is None:
            raise Exception("Optimizer has not been initialized yet. Please call 'initialize_optimizer' before proceeding to 'fit'.")

        # Copy input array to device
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)

        # Allocate output arrays - as torch tensors
        y_eq = torch.zeros((y.shape[-1] // self.samples_per_symbol, ), dtype=y.dtype).to(device=self.torch_device)

        # Learning rate scheduler (step lr - learning rate is changed every step)
        if self.lr_schedule:
            steps = 10
            reduction = 0.01
            gamma = np.exp(np.log(reduction) / steps)
            lr_schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=n_batches//steps, gamma=gamma)
        loss_curve = np.zeros((n_batches, ))

        # Loop over batches
        for n in range(n_batches):
            this_slice = slice(n * self.batch_size * self.samples_per_symbol,
                                n * self.batch_size * self.samples_per_symbol + self.batch_size * self.samples_per_symbol)

            xhat = self.forward(y[this_slice])

            loss = self._calculate_loss(xhat, y[this_slice])

            if n % self.loss_print_interval == 0:
                lrprint = lr_schedule.get_last_lr()[-1] if self.lr_schedule else self.learning_rate
                print(f"Batch {n}, Loss: {loss.item():.3f} (LR: {lrprint:.3e})")

            self._update_model(loss)

            if self.lr_schedule:
                lr_schedule.step()

            y_eq[n * self.batch_size: n * self.batch_size + self.batch_size] = xhat.clone().detach()
            loss_curve[n] = loss.item()

        return y_eq.detach().cpu().numpy(), loss_curve

    def apply(self, y_input):
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)
        with torch.set_grad_enabled(False):
            y_eq = self.forward(y)
        return y_eq.cpu().numpy()

    # weak implementation
    def get_parameters(self):
        raise NotImplementedError

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


class GenericTorchBlindProbabilisticEqualizer(GenericTorchBlindEqualizer):
    """
        Same as standard blind eqalizer, but with additional methods to calculate
        expectations and probabilistic output
    """
    IS_PROBABILISTIC = True

    def estimate_symbol_probs(self, y_input):
        y = torch.from_numpy(y_input).type(self.dtype).to(device=self.torch_device)
        with torch.set_grad_enabled(False):
            q_eq = self.forward_probs(y)
        return q_eq.cpu().numpy()

    # weak implementation
    def forward_probs(self, y_in):
        raise NotImplementedError


# Linear VAE
class VAELinearForward(GenericTorchBlindProbabilisticEqualizer):
    """
        Parent class for all the blind-equalizer VAEs with linear channel model
    """
    def __init__(self, channel_n_taps, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Channel model - in this type of VAE always a FIRfilter
        assert (samples_per_symbol <= channel_n_taps)
        self.channel_n_taps = channel_n_taps
        self.channel_filter = torch.zeros((self.channel_n_taps,), dtype=self.dtype)
        self.channel_filter.data[self.channel_n_taps // 2] = 1.0
        self.channel_filter = self.channel_filter.to(self.torch_device)
        self.channel_filter.requires_grad = True

        # Equaliser model - method initialize_decoder to be implemented by each child of this class
        self.equaliser = self.initialize_equaliser(**equaliser_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Process constellation
        self.constellation = torch.from_numpy(constellation).type(self.dtype).to(self.torch_device)
        self.constellation_size = len(self.constellation)

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / self.constellation_size

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def get_parameters(self):
        return [{'params': self.channel_filter, 'name': 'channel_filter'},
                {'params': self.equaliser.parameters(), 'name': 'equaliser'}]

    def initialize_equaliser(self, **equaliser_kwargs) -> torch.nn.Module:
        # weak implementation
        raise NotImplementedError

    def forward(self, y) -> torch.TensorType:
        # weak implementation
        raise NotImplementedError

    def forward_probs(self, y_in):
        y = self.forward(y_in)
        return self._soft_demapping(y)

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
        qest = self._soft_demapping(xhat)

        # KL Divergence term
        kl_div = torch.sum(qest[:, pre_delay:end_delay] * (torch.log(qest[:, pre_delay:end_delay] / self.constellation_prior[:, None] + self.epsilon)), dim=0)

        # Expectation of likelihood term - calculate subterms first - follow naming convention from Lauinger 2022
        expect_x = torch.zeros_like(y).to(self.torch_device)  # Insert zero on elements not in multiplum of SpS
        expect_x_sq = torch.zeros_like(y).to(self.torch_device)

        qc = qest * self.constellation[:, None]
        expect_x[::self.samples_per_symbol] = torch.sum(qc, dim=0)

        qc2 = qest * self.constellation[:, None] ** 2
        expect_x_sq[::self.samples_per_symbol] = torch.sum(qc2, dim=0)

        # More subterms - apply channel model to expected x
        h_ex = taf.convolve(expect_x, self.channel_filter, mode='valid')

        h_squared = torch.square(self.channel_filter)
        hsqconvx = taf.convolve(expect_x_sq - torch.square(expect_x), h_squared, mode='valid')
        e_hx2 = torch.sum(torch.square(h_ex) + hsqconvx)

        # Calculate loss - apply indexing to y to match with convolution
        exponent_term = torch.sum(torch.square(y[pre_delay:end_delay])) - 2 * torch.matmul(y[pre_delay:end_delay], h_ex) + e_hx2
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
        print("Equaliser")
        self.equaliser.print_parameters()
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
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_n_taps, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, torch_device, lr_schedule, **equaliser_kwargs)

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


class SecondVolterraVAE(VAELinearForward):
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_memory, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, torch_device, lr_schedule, **equaliser_kwargs)

    def initialize_equaliser(self, **equaliser_kwargs):
        # Equalizer is a second order Volterra model
        equaliser = SecondOrderVolterraSeries(n_lags1=equaliser_kwargs['equaliser_n_lags1'],
                                              n_lags2=equaliser_kwargs['equaliser_n_lags2'],
                                              samples_per_symbol=self.samples_per_symbol,
                                              dtype=self.dtype,
                                              torch_device=self.torch_device)
        return equaliser

    def forward(self, y):
        y_eq = self.equaliser.forward(y)
        return y_eq

    def __repr__(self) -> str:
        return "SecondVolterraVAE"


class VAESecondVolterraForward(GenericTorchBlindProbabilisticEqualizer):
    """
        Parent class for all the blind-equalizer VAEs with a Volterra channel model of order 2
    """
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True, noise_scaling=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Channel model - in this type of VAE always a Volterra filter of second order
        # FIXME: Currently assumes that n_taps equals lag in both 1st and 2nd order kernel
        assert (samples_per_symbol <= channel_memory)
        self.channel_memory = channel_memory
        channel_h1 = torch.zeros((self.channel_memory,), dtype=self.dtype)
        self.channel_h1 = channel_h1.to(self.torch_device)
        self.channel_h1.data[self.channel_memory // 2] = 1.0
        self.channel_h1.requires_grad = True

        # Initialize second order kernel
        channel_h2 = torch.zeros((self.channel_memory, self.channel_memory), dtype=self.dtype)
        self.channel_h2 = channel_h2.to(self.torch_device)
        self.channel_h2.requires_grad = True

        # Equaliser model - method initialize_decoder to be implemented by each child of this class
        self.equaliser = self.initialize_equaliser(**equaliser_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Learning parameters
        self.learning_rate_second_order = learning_rate # / 10  # define lr of second order kernel in channel model

        # Process constellation
        self.constellation = torch.from_numpy(constellation).type(self.dtype).to(self.torch_device)
        self.constellation_size = len(self.constellation)

        # Define noise scaling pr. symbol
        self.noise_scaling = torch.ones_like(self.constellation)
        self.learnable_noise_scaling = noise_scaling
        self.noise_scaling.requires_grad = True if self.learnable_noise_scaling else False

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / self.constellation_size

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def get_parameters(self):
        params = [{'params': self.channel_h1, 'name': 'channel_h1'},
                  {'params': self.channel_h2, "lr": self.learning_rate_second_order, 'name': 'channel_h2'},
                  {'params': self.equaliser.parameters(), 'name': 'equaliser'}
                ]

        if self.learnable_noise_scaling:
            params += [{'params': self.noise_scaling, 'name': 'noise_scale'}]

        return params

    def initialize_equaliser(self, **equaliser_kwargs) -> torch.nn.Module:
        # weak implementation
        raise NotImplementedError

    def forward(self, y) -> torch.TensorType:
        # weak implementation
        raise NotImplementedError

    def forward_probs(self, y_in):
        y = self.forward(y_in)
        return self._soft_demapping(y)

    def _soft_demapping(self, xhat):
        # Produce softmax outputs from equalised signal
        # xhat is a [N] tensor (vector of inputs)
        # output is [M, N] tensor, where M is the number of unique amplitude levels for the constellation
        # FIXME: Implement the real soft-demapping from (Lauinger, 2022) based on Maxwell-Boltzmann distribution
        # NB! Lauinger has a normalization step of xhat before this. Removed because it is not needed?
        qest = torch.transpose(self.sm.forward(-(torch.outer(xhat, torch.ones_like(self.constellation)) - self.constellation)**2 / (self.noise_variance * self.noise_scaling)), 1, 0)
        return qest

    def _unfold_and_flip(self, input_seq: torch.TensorType):
        # helper function to create the toeplitz matrix
        return torch.flip(input_seq.unfold(0, self.channel_memory, 1), (1, ))

    def _apply_second_order_kernel(self, xin: torch.TensorType, kernel: torch.TensorType):
        #Xlag = xin.unfold(0, kernel.shape[0], 1)  # use unfold to get Toeplitz matrix
        #Xlag = torch.flip(Xlag, (1, ))
        Xlag = self._unfold_and_flip(xin)
        Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)
        return torch.einsum('ijk,jk->i', Xouter, kernel)

    def _calculate_loss(self, xhat, y):
        pre_delay = (self.channel_memory - 1) // 2
        end_delay = -(self.channel_memory - 1) // 2

        # Symmetrizize second order kernel
        H = self.channel_h2 + self.channel_h2.T

        # Do soft-demapping on equalised signal
        qest = self._soft_demapping(xhat)

        # KL Divergence term
        kl_div = torch.sum(qest[:, pre_delay:end_delay] * (torch.log(qest[:, pre_delay:end_delay] / self.constellation_prior[:, None] + self.epsilon)), dim=0)

        # Expectation of likelihood term - calculate subterms first - follow naming convention from (Lauinger 2022, https://github.com/kit-cel/vae-equalizer)
        ex = torch.zeros_like(y).to(self.torch_device)  # Insert zero on elements not in multiplum of SpS
        ex2 = torch.zeros_like(y).to(self.torch_device)
        ex3 = torch.zeros_like(y).to(self.torch_device)
        ex4 = torch.zeros_like(y).to(self.torch_device)

        qc = qest * self.constellation[:, None]
        ex[::self.samples_per_symbol] = torch.sum(qc, dim=0)

        qc2 = qest * self.constellation[:, None] ** 2
        ex2[::self.samples_per_symbol] = torch.sum(qc2, dim=0)

        qc3 = qest * self.constellation[:, None] ** 3
        ex3[::self.samples_per_symbol] = torch.sum(qc3, dim=0)

        qc4 = qest * self.constellation[:, None] ** 4
        ex4[::self.samples_per_symbol] = torch.sum(qc4, dim=0)

        # More subterms - apply channel model to expected x
        h_ex = taf.convolve(ex, self.channel_h1, mode='valid')
        h2ex = self._apply_second_order_kernel(ex, H) + taf.convolve(ex2, torch.diag(self.channel_h2), mode='valid') - taf.convolve(ex**2, torch.diag(self.channel_h2), mode='valid')

        # Compute higher order interaction terms
        # E[(h * x)^2] - expectation of h applied to x squared
        h_squared = torch.square(self.channel_h1)
        hsqconvx = taf.convolve(ex2 - torch.square(ex), h_squared, mode='valid')
        e_hx2 = torch.sum(torch.square(h_ex) + hsqconvx)

        # Construct all the needed lag matrices for higher order interaction terms
        ex_lag = self._unfold_and_flip(ex)
        ex2_lag = self._unfold_and_flip(ex2)
        ex3_lag = self._unfold_and_flip(ex3)

        # E[(x h x H x) - h and H cross term with x
        ex2ex = torch.einsum('ni,nj->ij', ex2_lag, ex_lag)
        exsqex = torch.einsum('ni,nj->ij', ex_lag**2, ex_lag)
        x2x_sum = torch.sum(torch.multiply(ex2ex - exsqex,
                                           2 * torch.einsum('i,ji->ij', self.channel_h1, H) + torch.einsum('j,ii->ij', self.channel_h1, H)))
        x3_sum = torch.sum(taf.convolve(ex3 - 3 * ex2 * ex + 2*ex**3, self.channel_h1 * torch.diag(H), mode='valid'))

        e_xhxHx =  torch.sum(torch.einsum('ni,i->n', ex_lag, self.channel_h1) * torch.einsum('ni,nj,ij->n', ex_lag, ex_lag, H)) + x2x_sum + x3_sum

        # E[( x H x ^ 2)]  - expectation of H applied to x squared
        xxxx_sum = torch.sum(torch.einsum('ni,nj,ij->n', ex_lag, ex_lag, H)**2)

        ex2exex = torch.einsum('ni,nj,nk->ijk', ex2_lag, ex_lag, ex_lag)
        exsqexex = torch.einsum('ni,nj,nk->ijk', ex_lag**2, ex_lag, ex_lag)
        x2xx_sum = torch.sum(torch.multiply(ex2exex - exsqexex,
                                            2 * torch.einsum('ii,jk->ijk', H, H) + 4 * torch.einsum('ij,ik->ijk', H, H)))

        ex2ex2 = torch.einsum('ni,nj->ij', ex2_lag, ex2_lag)
        ex2exsq = torch.einsum('ni,nj->ij', ex2_lag, ex_lag**2)
        exsqex2 = torch.einsum('ni,nj->ij', ex_lag**2, ex2_lag)
        exsqexsq = torch.einsum('ni,nj->ij', ex_lag**2, ex_lag**2)
        x2x2_sum = torch.sum(torch.multiply(ex2ex2 - ex2exsq - exsqex2 + exsqexsq,
                                            2 * torch.einsum('ij,ij->ij', H, H) + torch.einsum('ii,jj->ij', H, H)))

        ex3ex = torch.einsum('ni,nj->ij', ex3_lag, ex_lag)
        ex2twoex = torch.einsum('ni,ni,nj->ij', ex2_lag, ex_lag, ex_lag)
        excubex = torch.einsum('ni,nj->ij', ex_lag**3, ex_lag)
        x3x_sum = torch.sum(torch.multiply(ex3ex - 3 * ex2twoex + 2 * excubex,
                                           4 * torch.einsum('ii,ij->ij', H, H)))

        x4_sum = torch.sum(taf.convolve(ex4 + 12 * ex2 * ex ** 2 - 3 * ex2 ** 2 - 4 * ex3 * ex - 6 * ex ** 4, torch.diag(H)**2, mode='valid'))

        e_Hx2 = xxxx_sum + x2xx_sum + x2x2_sum + x3x_sum + x4_sum

        # Calculate loss - apply indexing to y to remove boundary effects from convolution (mode='valid')
        y_times_filters = torch.matmul(y[pre_delay:end_delay], h_ex) + torch.matmul(y[pre_delay:end_delay], h2ex)
        exponent_term = torch.sum(torch.square(y[pre_delay:end_delay])) - 2 * y_times_filters + e_hx2 + 2 * e_xhxHx + e_Hx2
        loss = torch.sum(kl_div) + (y.shape[-1] - self.channel_memory + 1) * torch.log(exponent_term)

        if self.adaptive_noise_variance:
            with torch.no_grad():
                self.noise_variance = exponent_term / (y.shape[-1] - self.channel_memory + 1)

        return loss

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Zero gradients
        self.noise_variance.grad = None
        self.channel_h1.grad = None
        self.channel_h2.grad = None
        self.equaliser.zero_grad()
        if self.learnable_noise_scaling:
            self.noise_scaling.grad = None

    def print_model_parameters(self):
        # Print convolutional kernels
        print(f"Equaliser: {self.equaliser}")
        print(f"Channel (1st order): {self.channel_h1}")
        print(f"Channel (2nd order): {self.channel_h2}")

        # Print noise variance and scaling
        print(f"Noise variance: {self.noise_variance}")
        if self.learnable_noise_scaling:
            print(f"Noise scaling: {self.noise_scaling}")

    def train_mode(self):
        self.channel_h1.requires_grad = True
        self.channel_h2.requires_grad = True
        self.equaliser.train()

    def eval_mode(self):
        self.channel_h1.requires_grad = False
        self.channel_h2.requires_grad = False
        self.equaliser.eval()


class LinearV2VAE(VAESecondVolterraForward):
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_memory=channel_memory, learning_rate=learning_rate, constellation=constellation,
                         samples_per_symbol=samples_per_symbol, batch_size=batch_size, dtype=dtype,
                         noise_variance=noise_variance, adaptive_noise_variance=adaptive_noise_variance,
                         noise_scaling=True, torch_device=torch_device, lr_schedule=lr_schedule,
                         **equaliser_kwargs)

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
        return "LinV2VAE"


class SecondVolterraV2VAE(VAESecondVolterraForward):
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_memory=channel_memory, learning_rate=learning_rate, constellation=constellation,
                         samples_per_symbol=samples_per_symbol, batch_size=batch_size, dtype=dtype,
                         noise_variance=noise_variance, adaptive_noise_variance=adaptive_noise_variance,
                         noise_scaling=True, torch_device=torch_device, lr_schedule=lr_schedule,
                         **equaliser_kwargs)

    def initialize_equaliser(self, **equaliser_kwargs):
        # Equalizer is a second order Volterra model
        equaliser = SecondOrderVolterraSeries(n_lags1=equaliser_kwargs['equaliser_n_lags1'],
                                              n_lags2=equaliser_kwargs['equaliser_n_lags2'],
                                              samples_per_symbol=self.samples_per_symbol,
                                              dtype=self.dtype, torch_device=self.torch_device)
        return equaliser

    def forward(self, y):
        y_eq = self.equaliser.forward(y)
        return y_eq

    def __repr__(self) -> str:
        return "SecondVolterraV2VAE"


class SecondVolterraV2VAENoNoiseScaling(VAESecondVolterraForward):
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_memory=channel_memory, learning_rate=learning_rate, constellation=constellation,
                         samples_per_symbol=samples_per_symbol, batch_size=batch_size, dtype=dtype,
                         noise_variance=noise_variance, adaptive_noise_variance=adaptive_noise_variance,
                         noise_scaling=False, torch_device=torch_device, lr_schedule=lr_schedule,
                         **equaliser_kwargs)

    def initialize_equaliser(self, **equaliser_kwargs):
        # Equalizer is a second order Volterra model
        equaliser = SecondOrderVolterraSeries(n_lags1=equaliser_kwargs['equaliser_n_lags1'],
                                              n_lags2=equaliser_kwargs['equaliser_n_lags2'],
                                              samples_per_symbol=self.samples_per_symbol,
                                              dtype=self.dtype, torch_device=self.torch_device)
        return equaliser

    def forward(self, y):
        y_eq = self.equaliser.forward(y)
        return y_eq

    def __repr__(self) -> str:
        return "SecondVolterraV2VAE(Beta=1.0)"


class VAEHammersteinForward(GenericTorchBlindProbabilisticEqualizer):
    """
        Parent class for all the blind-equalizer VAEs with a Hammerstein channel model (static second order polynomial)
    """
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Channel model - in this type of VAE always a Volterra filter of second order
        assert (samples_per_symbol <= channel_memory)
        self.channel_memory = channel_memory
        channel_h = torch.zeros((self.channel_memory,), dtype=self.dtype)
        self.channel_h = channel_h.to(self.torch_device)
        self.channel_h.data[self.channel_memory // 2] = 1.0
        self.channel_h.requires_grad = True
        self.alpha_1 = torch.scalar_tensor(1.0, requires_grad=True)
        self.alpha_2 = torch.scalar_tensor(0.0, requires_grad=True)

        # Equaliser model - method initialize_decoder to be implemented by each child of this class
        self.equaliser = self.initialize_equaliser(**equaliser_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Learning parameters
        self.learning_rate_second_order = learning_rate # / 10  # define lr of second order kernel in channel model

        # Process constellation
        self.constellation = torch.from_numpy(constellation).type(self.dtype).to(self.torch_device)
        self.constellation_size = len(self.constellation)

        # Define noise scaling pr. symbol
        self.noise_scaling = torch.ones_like(self.constellation)
        self.noise_scaling.requires_grad = True

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / self.constellation_size

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def get_parameters(self):
        return [{'params': self.channel_h, 'name': 'channel_h'},
                {'params': self.alpha_1, 'name': 'alpha1'},
                {'params': self.alpha_2, 'name': 'alpha2'},
                {'params': self.equaliser.parameters(), 'name': 'equaliser'},
                {'params': self.noise_scaling, 'name': 'noise_scale'}]

    def initialize_equaliser(self, **equaliser_kwargs) -> torch.nn.Module:
        # weak implementation
        raise NotImplementedError

    def forward(self, y) -> torch.TensorType:
        # weak implementation
        raise NotImplementedError

    def forward_probs(self, y_in):
        y = self.forward(y_in)
        return self._soft_demapping(y)

    def _soft_demapping(self, xhat):
        # Produce softmax outputs from equalised signal
        # xhat is a [N] tensor (vector of inputs)
        # output is [M, N] tensor, where M is the number of unique amplitude levels for the constellation
        # FIXME: Implement the real soft-demapping from (Lauinger, 2022) based on Maxwell-Boltzmann distribution
        # NB! Lauinger has a normalization step of xhat before this. Removed because it is not needed?
        qest = torch.transpose(self.sm.forward(-(torch.outer(xhat, torch.ones_like(self.constellation)) - self.constellation)**2 / (self.noise_variance * self.noise_scaling)), 1, 0)
        return qest

    def _calculate_loss(self, xhat, y):
        pre_delay = (self.channel_memory - 1) // 2
        end_delay = -(self.channel_memory - 1) // 2

        # Do soft-demapping on equalised signal
        qest = self._soft_demapping(xhat)

        # KL Divergence term
        kl_div = torch.sum(qest[:, pre_delay:end_delay] * (torch.log(qest[:, pre_delay:end_delay] / self.constellation_prior[:, None] + self.epsilon)), dim=0)

        # Expectation of likelihood term - calculate subterms first - follow naming convention from (Lauinger 2022, https://github.com/kit-cel/vae-equalizer)
        ex = torch.zeros_like(y).to(self.torch_device)  # Insert zero on elements not in multiplum of SpS
        ex2 = torch.zeros_like(y).to(self.torch_device)
        ex3 = torch.zeros_like(y).to(self.torch_device)
        ex4 = torch.zeros_like(y).to(self.torch_device)

        qc = qest * self.constellation[:, None]
        ex[::self.samples_per_symbol] = torch.sum(qc, dim=0)

        qc2 = qest * self.constellation[:, None] ** 2
        ex2[::self.samples_per_symbol] = torch.sum(qc2, dim=0)

        qc3 = qest * self.constellation[:, None] ** 3
        ex3[::self.samples_per_symbol] = torch.sum(qc3, dim=0)

        qc4 = qest * self.constellation[:, None] ** 4
        ex4[::self.samples_per_symbol] = torch.sum(qc4, dim=0)

        # More subterms - apply channel model to expected x
        h_ex = self.alpha_1 * taf.convolve(ex, self.channel_h, mode='valid')
        h_ex2 = self.alpha_2 *taf.convolve(ex2, self.channel_h, mode='valid')

        # E[(h x h x^2) - cross term
        e_hxhx2 = torch.sum(taf.convolve(ex2, self.channel_h, mode='valid') * taf.convolve(ex, self.channel_h, mode='valid') + taf.convolve(ex3 - ex2 * ex, self.channel_h**2, mode='valid'))

        # E[(h x)^2]
        e_hxsq = torch.sum(taf.convolve(ex, self.channel_h, mode='valid')**2 + taf.convolve(ex2 - ex**2, self.channel_h**2, mode='valid'))

        # E[(h x^2)^2]
        e_hx2sq = torch.sum(taf.convolve(ex2, self.channel_h, mode='valid')**2 + taf.convolve(ex4 - ex2**2, self.channel_h**2, mode='valid'))

        # Calculate loss - apply indexing to y to remove boundary effects from convolution (mode='valid')
        y_times_filters = torch.matmul(y[pre_delay:end_delay], h_ex) + torch.matmul(y[pre_delay:end_delay], h_ex2)
        exponent_term = torch.sum(torch.square(y[pre_delay:end_delay])) - 2 * y_times_filters + 2 * self.alpha_1 * self.alpha_2 * e_hxhx2 + self.alpha_1**2 * e_hxsq + self.alpha_2**2 * e_hx2sq
        loss = torch.sum(kl_div) + (y.shape[-1] - self.channel_memory + 1) * torch.log(exponent_term)

        if self.adaptive_noise_variance:
            with torch.no_grad():
                self.noise_variance = exponent_term / (y.shape[-1] - self.channel_memory + 1)

        return loss

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Zero gradients
        self.noise_variance.grad = None
        self.channel_h.grad = None
        self.alpha_1.grad = None
        self.alpha_2.grad = None
        self.equaliser.zero_grad()

    def print_model_parameters(self):
        # Print convolutional kernels
        print(f"Equaliser: {self.equaliser}")
        print(f"Channel: {self.channel_h}")
        print(f"Channel (non-linearity): {self.alpha_1:.4f}x + {self.alpha_2:.4f}x^2")

        # Print noise variance
        print(f"Noise variance: {self.noise_variance}")

    def train_mode(self):
        self.channel_h.requires_grad = True
        self.alpha_1.requires_grad = True
        self.alpha_2.requires_grad = True
        self.equaliser.train()

    def eval_mode(self):
        self.channel_h.requires_grad = False
        self.alpha_1.requires_grad = False
        self.alpha_2.requires_grad = False
        self.equaliser.eval()


class SecondVolterraHVAE(VAEHammersteinForward):
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_memory, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, torch_device, lr_schedule, **equaliser_kwargs)

    def initialize_equaliser(self, **equaliser_kwargs):
        # Equalizer is a second order Volterra model
        equaliser = SecondOrderVolterraSeries(n_lags1=equaliser_kwargs['equaliser_n_lags1'],
                                              n_lags2=equaliser_kwargs['equaliser_n_lags2'],
                                              samples_per_symbol=self.samples_per_symbol,
                                              dtype=self.dtype, torch_device=self.torch_device)
        return equaliser

    def forward(self, y):
        y_eq = self.equaliser.forward(y)
        return y_eq

    def __repr__(self) -> str:
        return "SecondVolterraHVAE"


class VAEMemoryPolynomialForward(GenericTorchBlindProbabilisticEqualizer):
    """
        Parent class for all the blind-equalizer VAEs with a memory polynomial (MP) channel model
        Baseline model from Song et al (cf. appendix B). This is a simplified second order version
    """
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True, noise_scaling=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Channel model - in this type of VAE always a memory polynomial of second order
        assert (samples_per_symbol <= channel_memory)
        self.channel_memory = channel_memory
        channel_h1 = torch.zeros((self.channel_memory,), dtype=self.dtype)
        self.channel_h1 = channel_h1.to(self.torch_device)
        self.channel_h1.data[self.channel_memory // 2] = 1.0
        self.channel_h1.requires_grad = True

        # NB! Important for convergence that the second order term is initialized to zeros.
        channel_h2 = torch.zeros((self.channel_memory,), dtype=self.dtype)
        self.channel_h2 = channel_h2.to(self.torch_device)
        self.channel_h2.requires_grad = True

        # Equaliser model - method initialize_decoder to be implemented by each child of this class
        self.equaliser = self.initialize_equaliser(**equaliser_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Learning parameters
        self.learning_rate_second_order = learning_rate # / 10  # define lr of second order kernel in channel model

        # Process constellation
        self.constellation = torch.from_numpy(constellation).type(self.dtype).to(self.torch_device)
        self.constellation_size = len(self.constellation)

        # Define noise scaling pr. symbol
        self.noise_scaling = torch.ones_like(self.constellation)
        self.noise_scaling.requires_grad = noise_scaling

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / self.constellation_size

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def get_parameters(self):
        return [{'params': self.channel_h1, 'name': 'channel_h1'},
                {'params': self.channel_h2, 'name': 'channel_h2'},
                {'params': self.equaliser.parameters(), 'name': 'equaliser'},
                {'params': self.noise_scaling, 'name': 'noise_scale'}]

    def initialize_equaliser(self, **equaliser_kwargs) -> torch.nn.Module:
        # weak implementation
        raise NotImplementedError

    def forward(self, y) -> torch.TensorType:
        # weak implementation
        raise NotImplementedError

    def forward_probs(self, y_in):
        y = self.forward(y_in)
        return self._soft_demapping(y)

    def _soft_demapping(self, xhat):
        # Produce softmax outputs from equalised signal
        # xhat is a [N] tensor (vector of inputs)
        # output is [M, N] tensor, where M is the number of unique amplitude levels for the constellation
        # FIXME: Implement the real soft-demapping from (Lauinger, 2022) based on Maxwell-Boltzmann distribution
        # NB! Lauinger has a normalization step of xhat before this. Removed because it is not needed?
        qest = torch.transpose(self.sm.forward(-(torch.outer(xhat, torch.ones_like(self.constellation)) - self.constellation)**2 / (self.noise_variance * self.noise_scaling)), 1, 0)
        return qest

    def _calculate_loss(self, xhat, y):
        pre_delay = (self.channel_memory - 1) // 2
        end_delay = -(self.channel_memory - 1) // 2

        # Do soft-demapping on equalised signal
        qest = self._soft_demapping(xhat)

        # KL Divergence term
        kl_div = torch.sum(qest[:, pre_delay:end_delay] * (torch.log(qest[:, pre_delay:end_delay] / self.constellation_prior[:, None] + self.epsilon)), dim=0)

        # Expectation of likelihood term - calculate subterms first - follow naming convention from (Lauinger 2022, https://github.com/kit-cel/vae-equalizer)
        ex = torch.zeros_like(y).to(self.torch_device)  # Insert zero on elements not in multiplum of SpS
        ex2 = torch.zeros_like(y).to(self.torch_device)
        ex3 = torch.zeros_like(y).to(self.torch_device)
        ex4 = torch.zeros_like(y).to(self.torch_device)

        qc = qest * self.constellation[:, None]
        ex[::self.samples_per_symbol] = torch.sum(qc, dim=0)

        qc2 = qest * self.constellation[:, None] ** 2
        ex2[::self.samples_per_symbol] = torch.sum(qc2, dim=0)

        qc3 = qest * self.constellation[:, None] ** 3
        ex3[::self.samples_per_symbol] = torch.sum(qc3, dim=0)

        qc4 = qest * self.constellation[:, None] ** 4
        ex4[::self.samples_per_symbol] = torch.sum(qc4, dim=0)

        # Apply channel model to expected x
        h_ex = taf.convolve(ex, self.channel_h1, mode='valid')
        h_ex2 = taf.convolve(ex2, self.channel_h2, mode='valid')

        # E[(h1 x h2 x^2) - cross term
        e_hxhx2 = torch.sum(taf.convolve(ex2, self.channel_h2, mode='valid') * taf.convolve(ex, self.channel_h1, mode='valid') + taf.convolve(ex3 - ex2 * ex, self.channel_h1 * self.channel_h2, mode='valid'))

        # E[(h1 x)^2]
        e_hxsq = torch.sum(taf.convolve(ex, self.channel_h1, mode='valid')**2 + taf.convolve(ex2 - ex**2, self.channel_h1**2, mode='valid'))

        # E[(h2 x^2)^2]
        e_hx2sq = torch.sum(taf.convolve(ex2, self.channel_h2, mode='valid')**2 + taf.convolve(ex4 - ex2**2, self.channel_h2**2, mode='valid'))

        # Calculate loss - apply indexing to y to remove boundary effects from convolution (mode='valid')
        y_times_filters = torch.matmul(y[pre_delay:end_delay], h_ex) + torch.matmul(y[pre_delay:end_delay], h_ex2)
        exponent_term = torch.sum(torch.square(y[pre_delay:end_delay])) - 2 * y_times_filters + 2 * e_hxhx2 +  e_hxsq + e_hx2sq
        loss = torch.sum(kl_div) + (y.shape[-1] - self.channel_memory + 1) * torch.log(exponent_term)

        if self.adaptive_noise_variance:
            with torch.no_grad():
                self.noise_variance = exponent_term / (y.shape[-1] - self.channel_memory + 1)

        return loss

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Zero gradients
        self.noise_variance.grad = None
        self.channel_h1.grad = None
        self.channel_h2.grad = None
        self.equaliser.zero_grad()

    def print_model_parameters(self):
        # Print convolutional kernels
        print("Equaliser")
        self.equaliser.print_parameters()
        print(f"Channel (h1): {self.channel_h1}")
        print(f"Channel (h2): {self.channel_h2}")

        # Print noise variance
        print(f"Noise variance: {self.noise_variance}")

    def train_mode(self):
        self.channel_h1.requires_grad = True
        self.channel_h2.requires_grad = True
        self.equaliser.train()

    def eval_mode(self):
        self.channel_h1.requires_grad = False
        self.channel_h2.requires_grad = False
        self.equaliser.eval()


class SecondVolterraMPVAE(VAEMemoryPolynomialForward):
    """
        VAE with memory polynomial as channel model and second order Volterra equaliser
    """
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, noise_scaling=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(channel_memory, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, noise_scaling, torch_device, lr_schedule, **equaliser_kwargs)

    def initialize_equaliser(self, **equaliser_kwargs):
        # Equalizer is a second order Volterra model
        equaliser = SecondOrderVolterraSeries(n_lags1=equaliser_kwargs['equaliser_n_lags1'],
                                              n_lags2=equaliser_kwargs['equaliser_n_lags2'],
                                              samples_per_symbol=self.samples_per_symbol,
                                              dtype=self.dtype, torch_device=self.torch_device)
        return equaliser

    def forward(self, y):
        y_eq = self.equaliser.forward(y)
        return y_eq

    def __repr__(self) -> str:
        return "SecondVolterraMPVAE"

class VQVAESecondVolterraForward(GenericTorchBlindProbabilisticEqualizer):
    """
        Parent class for all the blind-equalizer VAEs with hard decoding with a Volterra channel model of order 2

        A special case of the memory polynomial VAE (MP-VAE) from

        J. Song et al., “Blind Channel Equalization Using Vector-Quantized Variational Autoencoders,”
          Feb. 22, 2023, arXiv: arXiv:2302.11687. doi: 10.48550/arXiv.2302.11687.

        where the general MP channel model has been replaced by a second order Volterra series

        Uses the "straight through"-trick to "differentiate" through the argmin operation.

    """
    def __init__(self, channel_memory, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True, noise_scaling=True,
                 torch_device=torch.device('cpu'), lr_schedule='step', **equaliser_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, learning_rate, dtype, torch_device, lr_schedule)

        # Channel model - in this type of VAE always a Volterra filter of second order
        # FIXME: Currently assumes that n_taps equals lag in both 1st and 2nd order kernel
        assert (samples_per_symbol <= channel_memory)
        self.channel_memory = channel_memory
        channel_h1 = torch.zeros((self.channel_memory,), dtype=self.dtype)
        self.channel_h1 = channel_h1.to(self.torch_device)
        self.channel_h1.data[self.channel_memory // 2] = 1.0
        self.channel_h1.requires_grad = True

        # Initialize second order kernel
        channel_h2 = torch.zeros((self.channel_memory, self.channel_memory), dtype=self.dtype)
        self.channel_h2 = channel_h2.to(self.torch_device)
        self.channel_h2.requires_grad = True

        # Equaliser model - method initialize_decoder to be implemented by each child of this class
        self.equaliser = self.initialize_equaliser(**equaliser_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Learning parameters
        self.learning_rate_second_order = learning_rate # / 10  # define lr of second order kernel in channel model

        # Process constellation
        self.constellation = torch.from_numpy(constellation).type(self.dtype).to(self.torch_device)
        self.constellation_size = len(self.constellation)

        # Define noise scaling pr. symbol
        self.noise_scaling = torch.ones_like(self.constellation)
        self.learnable_noise_scaling = noise_scaling
        self.noise_scaling.requires_grad = True if self.learnable_noise_scaling else False

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / self.constellation_size

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def get_parameters(self):
        params = [{'params': self.channel_h1, 'name': 'channel_h1'},
                  {'params': self.channel_h2, "lr": self.learning_rate_second_order, 'name': 'channel_h2'},
                  {'params': self.equaliser.parameters(), 'name': 'equaliser'}
                ]

        if self.learnable_noise_scaling:
            params += [{'params': self.noise_scaling, 'name': 'noise_scale'}]

        return params

    def initialize_equaliser(self, **equaliser_kwargs) -> torch.nn.Module:
        # weak implementation
        raise NotImplementedError

    def forward(self, y) -> torch.TensorType:
        # weak implementation
        raise NotImplementedError

    def forward_probs(self, y_in):
        y = self.forward(y_in)
        return self._hard_demapping(y)

    def _hard_demapping(self, xhat):
        # Find the closest constellation symbol and hard assign
        # Copies gradient through the argmin operation (StraightThrough)
        distances_sq = torch.square(xhat[:, None] - self.constellation[None, :])  # size [N x C]
        return torch.matmul(StraightThroughArgMin.apply(distances_sq), self.constellation)

    def _unfold_and_flip(self, input_seq: torch.TensorType):
        # helper function to create the toeplitz matrix
        return torch.flip(input_seq.unfold(0, self.channel_memory, 1), (1, ))

    def _apply_second_order_kernel(self, xin: torch.TensorType, kernel: torch.TensorType):
        #Xlag = xin.unfold(0, kernel.shape[0], 1)  # use unfold to get Toeplitz matrix
        #Xlag = torch.flip(Xlag, (1, ))
        Xlag = self._unfold_and_flip(xin)
        Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)
        return torch.einsum('ijk,jk->i', Xouter, kernel)

    def _calculate_loss(self, xhat, y):
        pre_delay = (self.channel_memory - 1) // 2
        end_delay = -(self.channel_memory - 1) // 2

        # Symmetrizize second order kernel
        H = self.channel_h2 + self.channel_h2.T

        # Hard assigment of of equalized signal to their nearest constellation point
        qest = self._hard_demapping(xhat)  # size N

        # FIXME: Implement the loss function from Song...
        loss = 0.0
        raise NotImplementedError


    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Zero gradients
        self.noise_variance.grad = None
        self.channel_h1.grad = None
        self.channel_h2.grad = None
        self.equaliser.zero_grad()
        if self.learnable_noise_scaling:
            self.noise_scaling.grad = None

    def print_model_parameters(self):
        # Print convolutional kernels
        print(f"Equaliser: {self.equaliser}")
        print(f"Channel (1st order): {self.channel_h1}")
        print(f"Channel (2nd order): {self.channel_h2}")

        # Print noise variance and scaling
        print(f"Noise variance: {self.noise_variance}")
        if self.learnable_noise_scaling:
            print(f"Noise scaling: {self.noise_scaling}")

    def train_mode(self):
        self.channel_h1.requires_grad = True
        self.channel_h2.requires_grad = True
        self.equaliser.train()

    def eval_mode(self):
        self.channel_h1.requires_grad = False
        self.channel_h2.requires_grad = False
        self.equaliser.eval()