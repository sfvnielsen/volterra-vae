import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as torchF
from itertools import combinations_with_replacement
from scipy.special import comb
from lib.utility import calculate_mmse_weights


class Passthrough(object):
    def __init__(self, samples_per_symbol) -> None:
        self.samples_per_symbol = samples_per_symbol
        self.needs_cpr = False

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
        return np.convolve(x, self.filter)[self.n_taps // 2:(self.n_taps // 2 + len(x))]

    def apply(self, x):
        return np.convolve(x, self.filter)[self.n_taps // 2:(self.n_taps // 2 + len(x))]

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
        self.needs_cpr = False  # default is that no carrier-phase recovery is needed

        assert (n_taps >= samples_per_symbol)

        # Initialize filter
        self.filter = np.zeros((n_taps,), dtype='complex128')
        self.filter[self.reference_tap] = 1.0 + 1j * 0.0

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
    def __init__(self, n_taps, learning_rate, learning_rate_phaseshift, samples_per_symbol=1, reference_tap=None) -> None:
        super().__init__(n_taps, learning_rate, samples_per_symbol, reference_tap)
        self.phase_shift = 0.0
        self.learning_rate_phaseshift = learning_rate_phaseshift

    def __repr__(self) -> str:
        return "LMSPilot"

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
            eq_out[i] = np.sum(delayline * self.filter * np.exp(-1j * self.phase_shift))
            error = eq_out[i] - tx_symbol[i]

            # Update filter and phaseshift
            self.filter = self.filter - self.learning_rate * np.conjugate(delayline) * error * np.exp(1j * self.phase_shift)
            self.phase_shift = self.phase_shift - self.learning_rate_phaseshift * np.imag(tx_symbol[i] * eq_out[i] * np.exp(-1j * self.phase_shift))

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


class VolterraPilot(object):
    """
        Naive implementation of the Volterra eqalizer with pilots.
    """

    def __init__(self, n_taps_pr_order: tuple, learning_rate: float, samples_per_symbol=1, dtype=np.complex128) -> None:
        self.order = len(n_taps_pr_order)
        self.n_taps_pr_order = n_taps_pr_order
        self.learning_rate = learning_rate
        self.samples_per_symbol = samples_per_symbol

        # Use formula from (Stojanovic, 2017) to calculate total number of taps
        self.total_taps = 0  # no dc coefficient
        self.indices_per_order = []
        for r, ntpo in enumerate(self.n_taps_pr_order):
            self.total_taps += comb(ntpo + r, r + 1)  # N-choose-k
            self.indices_per_order.append(list(combinations_with_replacement(range(ntpo), r+1)))
        self.total_taps = int(self.total_taps)

        # Assume that first order term has largest lag. Reference tap is 
        if len(n_taps_pr_order) > 1:
            assert np.argmax(n_taps_pr_order) == 0
        n_taps_linear = n_taps_pr_order[0]
        self.max_lag = n_taps_linear
        self.n_pad_start = n_taps_linear - n_taps_linear // 2 - 1  # zeropadding when ref tap central = int(np.floor(self.n_taps / 2))
        self.n_pad_end = n_taps_linear - self.n_pad_start

        self.filter = np.zeros((self.total_taps,), dtype=dtype)
        self.dc = 0.0
        print(f"Volterra equaliser with orders: {self.n_taps_pr_order} has {self.total_taps} free parameters.")

    def __repr__(self) -> str:
        return "VolterraPilot()"

    def _prep_volterra_lags(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # FIXME: Write this using np.take?
        xinput = []

        for o, ntpo in enumerate(self.n_taps_pr_order):
            # Fetch the indices
            indices_combinations = self.indices_per_order[o]

            # Create an empty array to store the subsequences
            subsequences = np.empty((len(indices_combinations), o+1), dtype=x.dtype)

            # Populate the array with the actual values from x
            for i, indices in enumerate(indices_combinations):
                subsequences[i] = x[-ntpo::][::-1][list(indices)]

            # Calculate the product of each row (subsequence)
            products = np.product(subsequences, axis=1)

            xinput.append(products)

        return np.concatenate(xinput)

    def fit(self, receiver_signal, tx_symbol):
        # Check input and figure out how many symbols that are being transmitted
        #n_symbols = self._check_input(receiver_signal)
        # FIXME: Account for samples pr. symbol
        #if n_symbols != len(tx_symbol):
        #    raise Exception(f"Number of receiver samples does not match with the number of supplied tx symbols!")

        # Allocate output arrays
        eq_out = np.zeros((len(tx_symbol), ), dtype=receiver_signal.dtype)

        # Zero pad input signal to account for delay to reference tap
        receiver_signal_padded = np.concatenate((np.zeros((self.n_pad_start,), dtype=receiver_signal.dtype),
                                                 receiver_signal,
                                                 np.zeros((self.n_pad_end,), dtype=receiver_signal.dtype)))

        # Loop over input signals
        for i, txi in enumerate(tx_symbol):
            # Get signal slice
            sigslice = slice(i * self.samples_per_symbol, i * self.samples_per_symbol + self.max_lag)

            # Equalizer output and error signal
            x_vol = self._prep_volterra_lags(receiver_signal_padded[sigslice])
            eq_out[i] = np.sum(x_vol * self.filter) + self.dc
            error = eq_out[i] - txi

            # Update filter and dc
            self.filter = self.filter - self.learning_rate * np.conjugate(x_vol) * error
            # self.dc = self.dc + self.learning_rate * np.mean(receiver_signal_padded[sigslice])

            # If filter has exploded, terminate
            if np.any(np.isnan(self.filter)):
                print('WARNING! Filter contains NaN. Terminating fit call.')
                break

        return eq_out

    def apply(self, receiver_signal):
        # Allocate output arrays
        # FIXME: Account for samples pr symbol
        eq_out = np.zeros((len(receiver_signal), ), dtype=receiver_signal.dtype)

        # Zero pad input signal to account for delay to reference tap
        receiver_signal_padded = np.concatenate((np.zeros((self.n_pad_start,), dtype=receiver_signal.dtype),
                                                 receiver_signal,
                                                 np.zeros((self.n_pad_end,), dtype=receiver_signal.dtype)))
        
        for i in range(0, len(receiver_signal)):
            # Get signal slice
            sigslice = slice(i * self.samples_per_symbol, i * self.samples_per_symbol + self.max_lag)

            # Equalizer output and error signal
            x_vol = self._prep_volterra_lags(receiver_signal_padded[sigslice])
            eq_out[i] = np.sum(x_vol * self.filter) + self.dc

        return eq_out


class CMA(Equalizer):
    # FIXME: Make a faster implementation that can utilize GPU
    # FIXME: Implement a batch'ed version
    def __init__(self, n_taps, learning_rate, constellation, dispersion_order=2,
                 samples_per_symbol=1, reference_tap=None) -> None:
        super().__init__(n_taps, learning_rate, samples_per_symbol, reference_tap)
        allowed_dispersion_orders = [1, 2]
        assert (dispersion_order in allowed_dispersion_orders)
        self.dispersion_order = dispersion_order
        self.constellation = constellation
        self.needs_cpr = True

        # Calculate needed summary statistics for the constellation (Rp from Godard 1980)
        self.optimization_target = np.divide(np.average(np.power(np.absolute(constellation), 2 * self.dispersion_order)),
                                             np.average(np.power(np.absolute(constellation), self.dispersion_order)))

        self.filter_update = self._filter_update_order2
        if self.dispersion_order == 1:
            self._filter_update_order1

    def __repr__(self) -> str:
        return "CMA"

    def _filter_update_order1(self, eq_out, delayline):
        return self.filter - self.learning_rate * np.conjugate(delayline) * eq_out * (1 - self.optimization_target / np.absolute(eq_out))

    def _filter_update_order2(self, eq_out, delayline):
        return self.filter - self.learning_rate * np.conjugate(delayline) * eq_out * (np.square(np.absolute(eq_out)) - self.optimization_target)

    def fit(self, receiver_signal):
        # Check input and figure out how many symbols that are being transmitted
        n_symbols = self._check_input(receiver_signal)
        eq_out = np.zeros((n_symbols, ), dtype=receiver_signal.dtype)

        # Zero pad input signal to account for delay to reference tap
        receiver_signal_padded = np.concatenate((np.zeros((self.n_pad_start,), dtype=receiver_signal.dtype),
                                                 receiver_signal,
                                                 np.zeros((self.n_pad_end,), dtype=receiver_signal.dtype)))

        # Loop over input signals
        for i in range(0, n_symbols):
            # Get signal slice
            sigslice = slice(i * self.samples_per_symbol, i * self.samples_per_symbol + self.n_taps)

            # Equalizer output
            delayline = np.flipud(receiver_signal_padded[sigslice])
            eq_out[i] = np.sum(delayline * self.filter)

            # Update filter
            self.filter = self.filter_update(eq_out[i], delayline)

            # If filter has exploded, terminate
            if np.any(np.isnan(self.filter)):
                print('WARNING! Filter contains NaN. Terminating fit call.')
                break

        return eq_out

    def apply(self, receiver_signal):
        # Check input and figure out how many symbols that are being transmitted
        n_symbols = self._check_input(receiver_signal)
        eq_out = np.zeros((n_symbols, ), dtype=receiver_signal.dtype)

        # Zero pad input signal to account for delay to reference tap
        receiver_signal_padded = np.concatenate((np.zeros((self.n_pad_start,), dtype=receiver_signal.dtype),
                                                 receiver_signal,
                                                 np.zeros((self.n_pad_end,), dtype=receiver_signal.dtype)))

        # Loop over input signals
        for i in range(0, n_symbols):
            # Get signal slice
            sigslice = slice(i * self.samples_per_symbol, i * self.samples_per_symbol + self.n_taps)

            # Equalizer output
            delayline = np.flipud(receiver_signal_padded[sigslice])
            eq_out[i] = np.sum(delayline * self.filter)

        return eq_out


class GenericTorchPilotEqualizer(object):
    """ Parent class that implements a supervised equalizer (with a pilot sequence)
    """
    def __init__(self, samples_per_symbol, batch_size, dtype=torch.float32, torch_device=torch.device("cpu"), flex_update_interval=None) -> None:
        # FIXME: "Flex" update scheme from (Lauinger, 2022)
        self.samples_per_symbol = samples_per_symbol
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.dtype = dtype
        self.complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128  # FIXME: weak check but works in most cases
        self.loss_print_interval = 1000  # FIXME: Make part of constructor
        conjugation_operator = torch.Tensor([1.0, -1.0])[:, None]
        self.conjugation_operator = conjugation_operator.to(self.torch_device)
        self.needs_cpr = False  # default is that no carrier-phase recovery is needed

    def _check_input(self, input_array, syms_array):
        n_batches_input = len(input_array) // self.samples_per_symbol // self.batch_size
        n_batches_syms = len(syms_array) // self.batch_size
        if n_batches_input * self.samples_per_symbol * self.batch_size != len(input_array):
            print(f"Warning! Number of samples in receiver signal does not match with a multiplum of the symbol length ({self.samples_per_symbol})")

        if n_batches_input != n_batches_syms:
            raise Exception(f"Number of supplied inputs batches ({n_batches_input}) does not match number symbol batches ({n_batches_syms})")

        return n_batches_input

    def to_real_stacked(self, x_cmplx_array):
        return np.concatenate((np.real(x_cmplx_array[np.newaxis, :]), np.imag(x_cmplx_array[np.newaxis, :])), axis=0)

    def to_complex(self, x_stacked_array):
        return np.squeeze(x_stacked_array[0, :] + 1j * x_stacked_array[1, :])

    def fit(self, y_input, tx_symbol):
        # Check input
        n_batches = self._check_input(y_input, tx_symbol)

        # Copy input and symbol arrays to device
        y = torch.from_numpy(self.to_real_stacked(y_input)).type(self.dtype).to(device=self.torch_device)
        tx_symbol = torch.from_numpy(self.to_real_stacked(tx_symbol)).type(self.dtype).to(device=self.torch_device)

        # Allocate output arrays - as torch tensors
        y_eq = torch.zeros((2, y.shape[-1] // self.samples_per_symbol), dtype=y.dtype).to(device=self.torch_device)

        # Loop over batches
        for n in range(n_batches):
            this_input_slice = slice(n * self.batch_size * self.samples_per_symbol,
                                     n * self.batch_size * self.samples_per_symbol + self.batch_size * self.samples_per_symbol)
            this_sym_slice = slice(n * self.batch_size,
                                   n * self.batch_size + self.batch_size)

            xhat = self.forward(y[:, this_input_slice])

            loss = self._calculate_loss(xhat, y[:, this_input_slice], tx_symbol[:, this_sym_slice])

            if n % self.loss_print_interval == 0:
                print(f"Batch {n}, Loss: {loss.item():.3f}")

            self._update_model(loss)

            y_eq[:, n * self.batch_size: n * self.batch_size + self.batch_size] = xhat.clone().detach()

        return self.to_complex(y_eq.detach().cpu().numpy())

    def apply(self, y_input):
        y = torch.from_numpy(self.to_real_stacked(y_input)).type(self.dtype).to(device=self.torch_device)
        with torch.set_grad_enabled(False):
            y_eq = self.forward(y)
        return self.to_complex(y_eq.cpu().numpy())

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
    def __init__(self, n_taps, reference_tap, learning_rate, samples_per_symbol,
                 batch_size, dtype=torch.float32, torch_device=torch.device("cpu"), flex_update_interval=None) -> None:
        super().__init__(samples_per_symbol, batch_size, dtype, torch_device, flex_update_interval)

        self.reference_tap = reference_tap
        self.learning_rate = learning_rate
        self.equaliser = torch.nn.Conv1d(kernel_size=n_taps, in_channels=2, out_channels=1,
                                         stride=self.samples_per_symbol, bias=False, padding=(n_taps - 1) // 2,
                                         dtype=self.dtype)
        torch.nn.init.dirac_(self.equaliser.weight)

         # Define optimizer object
        self.optimizer = torch.optim.SGD([{'params': self.equaliser.parameters()}],
                                          lr=self.learning_rate, momentum=0.0)

    def __repr__(self) -> str:
        return "LMSPilot(Torch)"

    def get_filter(self):
        return self.to_complex(self.equaliser.weight.squeeze().detach().cpu().numpy())

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.equaliser.zero_grad()

    def forward(self, y):
        y_eq = torch.empty((2, y.shape[-1] // self.samples_per_symbol))
        y_eq[0, :] = self.equaliser.forward(y[None, :, :]).squeeze()
        y_eq[1, :] = self.equaliser.forward((torch.roll(y, 1, dims=0) * self.conjugation_operator)[None, :, :]).squeeze()
        return y_eq

    def _calculate_loss(self, xhat, y, syms):
        # xhat and syms are [2, n_syms] tensors. Diff+square+sum = abs()**2
        return torch.mean(torch.sum(torch.square(xhat - syms), dim=0))

    def print_model_parameters(self):
        print(self.equaliser.weight)

    def train_mode(self):
        self.equaliser.train()

    def eval_mode(self):
        self.equaliser.eval()


class ComplexTanH(torch.nn.Module):
    """ Applies tanh to real and imaginary part of complex input
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        return self.tanh.forward(x.real) + 1j * self.tanh.forward(x.imag)


class TorchConvNNPilot(GenericTorchPilotEqualizer):
    """ Stacked Conv1D equaliser
    """
    def __init__(self, n_taps, learning_rate, samples_per_symbol, batch_size, n_conv_layers=2,
                 dtype=torch.float32, torch_device=torch.device("cpu"), flex_update_interval=None) -> None:
        super().__init__(samples_per_symbol, batch_size, dtype, torch_device, flex_update_interval)

        self.learning_rate = learning_rate

        # First n-1 conv latyers (without stride)
        self.conv_layers = torch.nn.ModuleList()
        for __ in range(n_conv_layers - 1):
            f = torch.nn.Conv1d(kernel_size=n_taps, in_channels=2, out_channels=2,
                                        bias=True, padding=(n_taps - 1) // 2,
                                        dtype=self.dtype)
            torch.nn.init.dirac_(f.weight)
            self.conv_layers.append(f)
            self.conv_layers.append(torch.nn.ELU())
            self.conv_layers.append(torch.nn.BatchNorm1d(num_features=2))

        # Last filter - with stride to only output one symbol
        self.last_filter = torch.nn.Conv1d(kernel_size=n_taps, in_channels=2, out_channels=2,
                                       stride=self.samples_per_symbol, bias=True, padding=(n_taps - 1) // 2,
                                       dtype=self.dtype)
        torch.nn.init.dirac_(self.last_filter.weight)

        # Fully connected to finish
        self.fc = torch.nn.Linear(in_features=2, out_features=2, bias=True)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.elu = torch.nn.ELU()

        # Define optimizer object
        self.optimizer = torch.optim.Adam([{'params': self.conv_layers.parameters()},
                                           {'params': self.last_filter.parameters()},
                                           {'params': self.fc.parameters()}],
                                          lr=self.learning_rate)

    def __repr__(self) -> str:
        return "ConvNN(Torch)"

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()
        for mod in [self.conv_layers, self.last_filter, self.fc]:
            mod.zero_grad()

    def forward(self, y):
        z = y[None, :, :]
        for lay in self.conv_layers:
            z = lay.forward(z)
        z2 = self.elu(self.last_filter.forward(z)).squeeze()
        y_eq = torch.permute(self.fc.forward(torch.permute(z2, (1,0))), (1,0))
        return y_eq

    def _calculate_loss(self, xhat, y, syms):
        # xhat and syms are [2, n_syms] tensors. Diff+square+sum = abs()**2
        return torch.mean(torch.sum(torch.square(xhat - syms), dim=0))

    def print_model_parameters(self):
        print("TODO: IMPLEMENT")

    def train_mode(self):
        for mod in [self.conv_layers, self.last_filter, self.fc]:
            mod.train()

    def eval_mode(self):
        for mod in [self.conv_layers, self.last_filter, self.fc]:
            mod.eval()


class ComplexFilter(torch.nn.Module):
    def __init__(self, n_taps: int, samples_per_symbol: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_taps = n_taps
        self.samples_per_symbol = samples_per_symbol
        init_filter = np.zeros((2, self.n_taps))
        init_filter[0, 0] = 1.0
        self.filter = torch.nn.Parameter(torch.from_numpy(init_filter), requires_grad=True)
        conjugation_operator = torch.Tensor([1.0, -1.0])[:, None]
        self.conjugation_operator = conjugation_operator.to("cpu")  # FIXME: Torch device?

    def forward(self, x):
        # input x assumed to be [timesteps, 2, n_taps]
        y_eq = torch.empty((2, x.shape[0]))
        y_eq[0, :] = torch.sum(self.filter[None, :, :] * x, dim=(1,2))
        y_eq[1, :] = torch.sum(self.filter[None, :, :] * torch.roll(x, 1, dims=1) * self.conjugation_operator, dim=(1,2))
        return y_eq


class TorchNvar(GenericTorchPilotEqualizer):
    """ Non-linear vector auto-regressive model
        cf. D. J. Gauthier et al, “Next generation reservoir computing,” 2021, doi: 10.1038/s41467-021-25801-2.
    """
    def __init__(self, order_in_symbols, learning_rate, samples_per_symbol,
                 batch_size, dtype=torch.float32, torch_device=torch.device("cpu"), flex_update_interval=None) -> None:
        super().__init__(samples_per_symbol, batch_size, dtype, torch_device, flex_update_interval)


        self.order_times_sps = order_in_symbols * samples_per_symbol
        odd_length = self.order_times_sps % 2
        linear_terms = self.order_times_sps
        non_linear_terms = self.order_times_sps * (self.order_times_sps + 1) / 2
        self.n_taps = int(linear_terms + non_linear_terms)
        print(f'Total number of taps in NVAR: {linear_terms + non_linear_terms}')

        # Define filter as a matrix (2 by number of taps)
        self.filter = ComplexFilter(self.n_taps, self.samples_per_symbol)
        self.padding = ((self.order_times_sps - 1) // 2, (self.order_times_sps - odd_length) // 2)

        # Define optimizer object
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD([{'params': self.filter.parameters()}],
                                          lr=self.learning_rate,
                                          momentum=0.0)

    def __repr__(self) -> str:
        return "NVAR(Torch)"

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.filter.zero_grad()

    def forward(self, y):
        # Pad with zeros
        ypadded = torchF.pad(y, self.padding, mode='constant', value=0.0)

        xfolded = ypadded.unfold(1, self.order_times_sps, self.samples_per_symbol)  # dims: [2, timesteps, SPS*MEMORY]
        xouter_re = torch.einsum('ij,ik->ijk', xfolded[0, :], xfolded[0, :]) - torch.einsum('ij,ik->ijk', xfolded[1, :], xfolded[1, :])  # dims: [2, timesteps, SPS*MEMORY, SPS*MEMORY]
        xouter_im = 2 * torch.einsum('ij,ik->ijk', xfolded[0, :], xfolded[1, :])  # dims: [2, timesteps, SPS*MEMORY, SPS*MEMORY]

        # Extract upper triangular part of all the submatries in the last two dimensions
        # then stack together with linear part
        triu_mask = torch.triu(torch.ones(self.order_times_sps, self.order_times_sps, dtype=torch.bool), diagonal=0).view(1, self.order_times_sps, self.order_times_sps)
        x_real_features = torch.concat((xfolded[0, :], xouter_re.masked_select(triu_mask).view(xouter_re.shape[0], -1)), dim=1)
        x_imag_features = torch.concat((xfolded[1, :], xouter_im.masked_select(triu_mask).view(xouter_im.shape[0], -1)), dim=1)
        xvec_tensor = torch.stack((x_real_features, x_imag_features), dim=1)

        # Apply filter
        y_eq = self.filter.forward(xvec_tensor)

        return y_eq

    def _calculate_loss(self, xhat, y, syms):
        # xhat and syms are [2, n_syms] tensors. Diff+square+sum = abs()**2
        return torch.mean(torch.sum(torch.square(xhat - syms), dim=0))

    def print_model_parameters(self):
        print("TODO: IMPLEMENT")

    def train_mode(self):
        self.filter.train()

    def eval_mode(self):
        self.filter.eval()


class GenericTorchBlindEqualizer(object):
    def __init__(self, samples_per_symbol, batch_size, dtype=torch.float32, torch_device=torch.device("cpu"), flex_update_interval=None) -> None:
        # FIXME: "Flex" update scheme from (Lauinger, 2022)
        self.samples_per_symbol = samples_per_symbol
        self.batch_size = batch_size
        self.torch_device = torch_device
        self.dtype = dtype
        self.complex_dtype = torch.complex64 if self.dtype == torch.float32 else torch.complex128  # FIXME: weak check but works in most cases
        self.loss_print_interval = 100  # FIXME: Make part of constructor
        conjugation_operator = torch.Tensor([1.0, -1.0])[:, None]
        self.conjugation_operator = conjugation_operator.to(self.torch_device)
        self.needs_cpr = False  # default is that no carrier-phase recovery is needed

    def _check_input(self, input_array):
        n_batches = len(input_array) // self.samples_per_symbol // self.batch_size
        if n_batches * self.samples_per_symbol * self.batch_size != len(input_array):
            print(f"Warning! Number of samples in receiver signal does not match with a multiplum of the symbol length ({self.samples_per_symbol})")
        return n_batches

    def to_real_stacked(self, x_cmplx_array):
        return np.concatenate((np.real(x_cmplx_array[np.newaxis, :]), np.imag(x_cmplx_array[np.newaxis, :])), axis=0)

    def to_complex(self, x_stacked_array):
        return np.squeeze(x_stacked_array[0, :] + 1j * x_stacked_array[1, :])

    def fit(self, y_input):
        # Check input
        n_batches = self._check_input(y_input)

        # Copy input array to device
        y = torch.from_numpy(self.to_real_stacked(y_input)).type(self.dtype).to(device=self.torch_device)

        # Allocate output arrays - as torch tensors
        y_eq = torch.zeros((2, y.shape[-1] // self.samples_per_symbol), dtype=y.dtype).to(device=self.torch_device)

        # Loop over batches
        for n in range(n_batches):
            this_slice = slice(n * self.batch_size * self.samples_per_symbol,
                               n * self.batch_size * self.samples_per_symbol + self.batch_size * self.samples_per_symbol)

            xhat = self.forward(y[:, this_slice])

            loss = self._calculate_loss(xhat, y[:, this_slice])

            if n % self.loss_print_interval == 0:
                print(f"Batch {n}, Loss: {loss.item():.3f}")

            self._update_model(loss)

            y_eq[:, n * self.batch_size: n * self.batch_size + self.batch_size] = xhat.clone().detach()

        return self.to_complex(y_eq.detach().cpu().numpy())

    def apply(self, y_input):
        y = torch.from_numpy(self.to_real_stacked(y_input)).type(self.dtype).to(device=self.torch_device)
        with torch.set_grad_enabled(False):
            y_eq = self.forward(y)
        return self.to_complex(y_eq.cpu().numpy())

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


class VAELinearForward(GenericTorchBlindEqualizer):
    # Parent class for all the VAEs with linear forward model
    def __init__(self, encoder_n_taps, learning_rate, constellation: np.ndarray,
                 samples_per_symbol, batch_size, dtype=torch.float32, noise_variance=1.0,
                 adaptive_noise_variance=True,
                 torch_device=torch.device('cpu'), flex_update_interval=None, **decoder_kwargs) -> None:
        super().__init__(samples_per_symbol, batch_size, dtype, torch_device, flex_update_interval)

        # Encoder - n_taps (forward channel model)
        # NB! No padding applied to foward model
        assert (samples_per_symbol <= encoder_n_taps)
        self.encoder_n_taps = encoder_n_taps
        self.encoder = torch.nn.Conv1d(kernel_size=encoder_n_taps, in_channels=2, out_channels=1,
                                       bias=False, dtype=self.dtype)
        torch.nn.init.dirac_(self.encoder.weight)
        self.encoder.to(self.torch_device)

        # Decoder - equaliser model
        self.decoder = self.initialize_decoder(**decoder_kwargs)

        # Noise variance
        self.noise_variance = torch.scalar_tensor(noise_variance, dtype=self.dtype, requires_grad=False)
        self.noise_variance.to(self.torch_device)
        self.adaptive_noise_variance = adaptive_noise_variance

        # Learning parameters
        self.learning_rate = learning_rate

        # Process constellation
        constellation_tensor = torch.from_numpy(constellation).type(self.complex_dtype).to(self.torch_device)
        self.constellation_real = torch.unique(torch.real(constellation_tensor))
        self.constellation_imag = torch.unique(torch.imag(constellation_tensor))
        assert (len(self.constellation_real) == len(self.constellation_imag))
        self.constellation_size = len(self.constellation_real) + len(self.constellation_imag)
        self.constellation_amplitudes = torch.concatenate((self.constellation_real, self.constellation_imag))

        # Define constellation prior
        # FIXME: Currently uniform - change in accordance to (Lauinger, 2022) (PCS)
        self.constellation_prior = torch.ones((self.constellation_size,), dtype=self.dtype, device=self.torch_device) / (self.constellation_size / 2)

        # Define optimizer object
        self.optimizer = torch.optim.Adam([{'params': self.encoder.parameters()},
                                           {'params': self.decoder.parameters()}],
                                          lr=self.learning_rate, amsgrad=True)

        # Define Softmax layer
        self.sm = torch.nn.Softmax(dim=-1)

        # Epsilon for log conversions
        self.epsilon = 1e-12

    def initialize_decoder(self, **decoder_kwargs):
        # weak implementation
        raise NotImplementedError

    def forward(self, y):
        # weak implementation
        raise NotImplementedError

    def _soft_demapping(self, xhat):
        # Produce softmax outputs from equalised signal
        # xhat is a [2, N] tensor (real imag stacked)
        # output is [2M, N] tensor, where M is the number of unique amplitude levels for the constellation
        qest = torch.empty((self.constellation_size, xhat.shape[-1]), dtype=self.dtype, device=self.torch_device)
        xn_real = xhat[0, :] / torch.abs(xhat[0, :]).mean() * torch.mean(torch.abs(self.constellation_real))
        xn_imag = xhat[1, :] / torch.abs(xhat[1, :]).mean() * torch.mean(torch.abs(self.constellation_imag))
        qest[0:self.constellation_size // 2, :] = torch.transpose(self.sm.forward(-(torch.outer(xn_real, torch.ones_like(self.constellation_real)) - self.constellation_real)**2 / (self.noise_variance)), 1, 0)
        qest[self.constellation_size // 2:, :] = torch.transpose(self.sm.forward(-(torch.outer(xn_imag, torch.ones_like(self.constellation_imag)) - self.constellation_imag)**2 / (self.noise_variance)), 1, 0)
        return qest

    def _calculate_loss(self, xhat, y):
        pre_delay = (self.encoder_n_taps - 1) // 2
        end_delay = -(self.encoder_n_taps - 1) // 2

        # Do soft-demapping on equalised signal
        # FIXME: How to handle models that output q-est directly?
        qest = self._soft_demapping(xhat)

        # KL Divergence term
        kl_div = torch.sum(qest[:, pre_delay:end_delay] * (torch.log(qest[:, pre_delay:end_delay] / self.constellation_prior[:, None] + self.epsilon)), dim=0)

        # Expectation of likelihood term - calculate subterms first - follow naming convention from Lauinger 2022
        expect_x = torch.zeros_like(y)  # FIXME: Insert zero on elements not in multiplum of SpS
        expect_x_sq = torch.zeros_like(y)

        qc = qest * self.constellation_amplitudes[:, None]
        expect_x[0, ::self.samples_per_symbol] = torch.sum(qc[0:self.constellation_size // 2, :], dim=0)
        expect_x[1, ::self.samples_per_symbol] = torch.sum(qc[self.constellation_size // 2:, :], dim=0)

        qc2 = qest * self.constellation_amplitudes[:, None] ** 2
        expect_x_sq[0, ::self.samples_per_symbol] = torch.sum(qc2[0:self.constellation_size // 2, :], dim=0)
        expect_x_sq[1, ::self.samples_per_symbol] = torch.sum(qc2[self.constellation_size // 2:, :], dim=0)

        # More subterms...
        expect_x_conj = expect_x * self.conjugation_operator
        d_real = self.encoder.forward(expect_x_conj[None, :, :]).squeeze()
        expect_x_iq_flipped = torch.roll(expect_x, 1, dims=0)  # flipping real and imaginary part
        d_imag = self.encoder.forward(expect_x_iq_flipped[None, :, :]).squeeze()

        h_squared = torch.sum(torch.square(self.encoder.weight), dim=1, keepdim=True)
        hsqconvx = torch.conv1d((torch.sum(expect_x_sq, dim=0) - torch.sum(expect_x**2, dim=0))[None, None, :], h_squared)
        hsqconvx = hsqconvx.squeeze()
        e_hx2 = torch.sum(torch.square(d_real) + torch.square(d_imag) + hsqconvx)

        # Calculate loss - apply indexing to y to match with convolution
        exponent_term = torch.sum(torch.square(y[:, pre_delay:end_delay])) - 2 * torch.matmul(y[0, pre_delay:end_delay], d_real) - 2 * torch.matmul(y[1, pre_delay:end_delay], d_imag) + e_hx2
        loss = torch.sum(kl_div) + (y.shape[-1] - self.encoder_n_taps + 1) * torch.log(exponent_term)

        if self.adaptive_noise_variance:
            with torch.no_grad():
                self.noise_variance = exponent_term / (y.shape[-1] - self.encoder_n_taps + 1)

        return loss

    def _update_model(self, loss):
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Zero gradients
        self.noise_variance.grad = None
        for module in [self.encoder, self.decoder]:
            module.zero_grad()

    def print_model_parameters(self):
        # Print convolutional kernels
        for module_name, module in zip(["encoder", "decoder"],
                                       [self.encoder, self.decoder]):
            print(f"{module_name}: {module.weight}")

        # Print noise variance
        print(f"Noise variance: {self.noise_variance}")

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()


class LinearVAE(VAELinearForward):
    def __init__(self, encoder_n_taps, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, torch_device=torch.device('cpu'), flex_update_interval=None, **decoder_kwargs) -> None:
        super().__init__(encoder_n_taps, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, torch_device, flex_update_interval, **decoder_kwargs)

    def initialize_decoder(self, **decoder_kwargs):
        # Decoder - n_taps (equaliser) (padded with zeros)
        decoder = torch.nn.Conv1d(kernel_size=decoder_kwargs['decoder_n_taps'], in_channels=2, out_channels=1,
                                  stride=self.samples_per_symbol, bias=False, padding=(decoder_kwargs['decoder_n_taps'] - 1) // 2,
                                  dtype=self.dtype)
        torch.nn.init.dirac_(decoder.weight)
        decoder.to(self.torch_device)
        return decoder

    def forward(self, y):
        # NB! In this formulation the weights are preconjugated
        y_eq = torch.empty((2, y.shape[-1] // self.samples_per_symbol))
        y_eq[0, :] = self.decoder.forward(y[None, :, :]).squeeze()
        y_eq[1, :] = self.decoder.forward((torch.roll(y, 1, dims=0) * self.conjugation_operator)[None, :, :]).squeeze()
        return y_eq

    def __repr__(self) -> str:
        return "LinVAE"

class PermuteLayer(torch.nn.Module):
    def __init__(self, perm: tuple, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.perm = perm

    def forward(self, x):
        return torch.permute(x, self.perm)


class Conv1DVAE(VAELinearForward):
    def __init__(self, encoder_n_taps, learning_rate, constellation: np.ndarray, samples_per_symbol,
                 batch_size, dtype=torch.float32, noise_variance=1, adaptive_noise_variance=True, torch_device=torch.device('cpu'), flex_update_interval=None, **decoder_kwargs) -> None:
        super().__init__(encoder_n_taps, learning_rate, constellation, samples_per_symbol,
                         batch_size, dtype, noise_variance, adaptive_noise_variance, torch_device, flex_update_interval, **decoder_kwargs)

    def initialize_decoder(self, decoder_n_taps, n_layers):
        # FIXME: Doesn't fit....
        # n conv1d layers with relus between. ff linear at the end
        decoder = torch.nn.Sequential()
        for __ in range(n_layers - 1):
            decoder.append(torch.nn.Conv1d(kernel_size=decoder_n_taps, in_channels=2, out_channels=2,
                                           bias=True, padding=(decoder_n_taps - 1) // 2,
                                           dtype=self.dtype))
            decoder.append(torch.nn.ELU())
            decoder.append(torch.nn.BatchNorm1d(num_features=2))

        # Add another Conv1d with stride - achieves desired samples_per_symbol
        decoder.append(torch.nn.Conv1d(kernel_size=decoder_n_taps, in_channels=2, out_channels=2,
                                       bias=True, padding=(decoder_n_taps - 1) // 2, stride=self.samples_per_symbol,
                                       dtype=self.dtype))
        decoder.append(torch.nn.ELU())
        decoder.append(torch.nn.BatchNorm1d(num_features=2))

        # Add linear layer in the end
        decoder.append(PermuteLayer((0, 2, 1)))
        decoder.append(torch.nn.Linear(in_features=2, out_features=2, bias=True, dtype=self.dtype))
        decoder.append(PermuteLayer((0, 2, 1)))

        # Move to device and return
        decoder.to(self.torch_device)
        return decoder

    def forward(self, y):
        y_eq = torch.empty((2, y.shape[-1] // self.samples_per_symbol))
        y_eq = self.decoder.forward(y[None, :, :]).squeeze()
        return y_eq

    def __repr__(self) -> str:
        return "StackedConv1DVAE"
