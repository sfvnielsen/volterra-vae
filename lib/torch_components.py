"""
    Module containing common components used in conjuction with the equalizer library
    All components are implemented in pytorch
"""


import numpy as np
import torch
import torch.nn.functional as F


class SecondOrderVolterraSeries(torch.nn.Module):
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
