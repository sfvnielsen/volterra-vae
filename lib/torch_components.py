"""
    Module containing common components used in conjuction with the equalizer library
    All components are implemented in pytorch
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout


class SecondOrderVolterraSeries(torch.nn.Module):
    """
        Vanilla second order Volterra-series implemented in torch using einsum
    """
    def __init__(self, n_lags1, n_lags2, samples_per_symbol, dtype=torch.double,
                 torch_device=torch.device('cpu'),
                 **kwargs) -> None:
        super(SecondOrderVolterraSeries, self).__init__(**kwargs)

        self.sps = samples_per_symbol
        self.dtype = dtype
        self.torch_device = torch_device
        assert (n_lags1 + 1) % 2 == 0  # assert odd lengths
        assert (n_lags2 + 1) % 2 == 0

        # Initialize first order kernel
        kernel1_init = np.zeros((n_lags1, ))
        kernel1_init[n_lags1 // 2] = 1.0  # dirac initialization
        self.kernel1 = torch.nn.Parameter(torch.from_numpy(kernel1_init).to(dtype).to(self.torch_device), requires_grad=True)
        self.lag1_padding = n_lags1 // 2

        # Initialize second order kernel (zeros)
        kernel2_init = torch.zeros((n_lags2, n_lags2), dtype=dtype)
        self.kernel2 = torch.nn.Parameter(kernel2_init.to(self.torch_device))
        self.lag2_padding = n_lags2 // 2

    def forward(self, x: torch.TensorType):
        # Output of first order kernel
        xpad = torch.concatenate((torch.zeros((self.lag1_padding), dtype=self.dtype, device=self.torch_device),
                                  x,
                                  torch.zeros((self.lag1_padding), dtype=self.dtype, device=self.torch_device)))
        y = F.conv1d(xpad[None, None, :], self.kernel1[None, None, :], stride=self.sps).squeeze()

        # Create lag and calculate the pr. lag outer product
        x2pad = torch.concatenate((torch.zeros((self.lag2_padding), dtype=self.dtype, device=self.torch_device),
                                   x,
                                   torch.zeros((self.lag2_padding), dtype=self.dtype, device=self.torch_device)))
        Xlag = torch.flip(x2pad.unfold(0, self.kernel2.shape[0], self.sps), (1,))
        Xouter = torch.einsum('ij,ik->ijk', Xlag, Xlag)

        # Apply kernel to outer prodcts pr lag - symmetrize kernel
        y2 = torch.einsum('ijk,jk->i', Xouter, self.kernel2 + self.kernel2.T)

        return y + y2


def universal_nonlinear_unit(hidden_size, n_layers, dropout_rate=0.1):
    """
        Helper function for creating a simple non-linear module
        Input (1D) -> Linear(1, M) -> ReLU -> 
            [ Linear(M, M) -> ReLU -> Dropout ] x n_layers
            -> Linear(M, 1) -> Out
    """
    layers = [Linear(in_features=1, out_features=hidden_size), ReLU()]
    for __ in range(n_layers):
        layers.append(Linear(in_features=hidden_size, out_features=hidden_size))
        layers.append(ReLU())
        layers.append(Dropout(dropout_rate))
    layers.append(Linear(in_features=hidden_size, out_features=1))
    return Sequential(*layers)


class WienerHammersteinNN(torch.nn.Module):
    """
        Neural network mimicking a Wiener-Hammerstein structure
        FIR -> NN modules -> FIR
        Inspired by equaliser from (Caciularu and Burshtein, 2020)
    """
    def __init__(self, n_lags, n_hidden_unlus, unlu_depth, unlu_hidden_size, samples_per_symbol, dtype=torch.double,
                 torch_device=torch.device('cpu'), **kwargs) -> None:
        super(WienerHammersteinNN, self).__init__(**kwargs)

        # Set properties of class
        self.sps = samples_per_symbol
        self.dtype = dtype
        self.torch_device = torch_device

        # Initialize kernels (FIR filters pre- and post non-linearity)
        assert (n_lags + 1) % 2 == 0
        kernel_init = np.zeros((n_lags, ))
        kernel_init[n_lags // 2] = 1.0  # dirac initialization
        self.kernel1 = torch.nn.Parameter(torch.from_numpy(kernel_init).to(dtype).to(self.torch_device))
        self.kernel2 = torch.nn.Parameter(torch.from_numpy(kernel_init).to(dtype).to(self.torch_device))
        self.padding = n_lags // 2

        # Initialize the universal nonlinear units (UNLU) - join them by a Linear layer
        self.unlus = torch.nn.ModuleList([universal_nonlinear_unit(unlu_hidden_size, unlu_depth).to(dtype).to(self.torch_device) for __ in range(n_hidden_unlus)])
        self.linear_join = Linear(in_features=n_hidden_unlus, out_features=1).to(dtype).to(self.torch_device)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # Apply first FIR
        xpad = torch.concatenate((torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device),
                                  x,
                                  torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device)))
        z = F.conv1d(xpad[None, None, :], self.kernel1[None, None, :]).squeeze()

        # Apply UNLUs
        u = []
        for un in self.unlus:
            u.append(un.forward(z[:, None]))
        
        # Join using linear layers
        z2 = self.linear_join.forward(torch.concatenate(u, dim=1)).squeeze()

        # Pad and run second FIR
        z2pad = torch.concatenate((torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device),
                                  z2,
                                  torch.zeros((self.padding), dtype=self.dtype, device=self.torch_device)))

        y = F.conv1d(z2pad[None, None, :], self.kernel2[None, None, :], stride=self.sps).squeeze()

        return y



