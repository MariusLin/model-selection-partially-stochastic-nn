import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Initialization.DWF_initlialization import dwf_initialization
import Utilities.activation_functions as af

"""
It defines a 2D convolutional layer where every parameter has a Gaussian prior
Additionally, we perform DWF on the standard deviations
"""
class FactorizedGaussianConv2dReparameterization(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, W_std=None, b_std=None,
                 prior_per="layer", scaled_variance=True, device = "cpu",  D = 3):
        super(FactorizedGaussianConv2dReparameterization, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.scaled_variance = scaled_variance
        self.D = D
        self.device = device

        if W_std is None:
            if self.scaled_variance:
                W_std = 1.
            else:
                W_std = 1. / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if b_std is None:
            b_std = 1.

        # Initialize the parameters
        if prior_per == "layer":
            W_shape, b_shape = (1), (1)
        elif prior_per == "parameter":
            W_shape = (self.out_channels, self.in_channels, *self.kernel_size)
            b_shape = (self.out_channels)

        self.W_mu = 0.
        self.b_mu = 0.
        self.W_omega_std = nn.ParameterList()
        # DWF Initialization for weight 
        omega_factors_W_std = dwf_initialization(
            [self.out_channels * self.in_channels * self.kernel_size[0]*self.kernel_size[1]], [W_std], self.D, W_std*0.25, 
            device = self.device)[0]  # shape (layer_size, D)
        # Store D separate factor tensors
        for d in range(self.D):
            param = nn.Parameter(omega_factors_W_std[:, d].view(*W_shape), requires_grad=True) 
            self.W_omega_std.append(param)
        
        if self.bias:
            self.b_omega_std = nn.ParameterList()
            omega_factors_b_std = dwf_initialization(
            [self.out_channels], [b_std], self.D, b_std*0.25, device = self.device)[0]  # shape (layer_size, D)
            # Store D separate factor tensors
            for d in range(self.D):
                param = nn.Parameter(omega_factors_b_std[:, d].view((b_shape, )), requires_grad=True) 
                self.b_omega_std.append(param)
        else:
            self.register_buffer(
                'b_std', torch.ones(b_shape))

    def forward(self, X):
        W_std = self.get_W_std()
        W = self.W_mu + af.adapted_softplus(W_std) *\
            torch.randn((self.out_channels, self.in_channels,
                         *self.kernel_size), device=self.device)
        if self.scaled_variance:
            W = W / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias:
            b_std = self.get_b_std()
            b = self.b_mu + af.adapted_softplus(b_std) *\
                torch.randn((self.out_channels), device=self.device)
        else:
            b = torch.zeros((self.out_channels), device=self.device)
        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)
    
    def get_W_std(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the weight matrices elementwise
        """
        W_std = self.W_omega_std[0]
        for d in range(1, self.D):
            W_std = W_std * self.W_omega_std[d]
        mask = torch.abs(W_std) < eps_tiny
        W_std = W_std.masked_fill(mask, 0.0)
        return W_std

    def get_b_std(self, eps_tiny = 1.19e-7):
        """
        Multiplies all the bias vectors elementwise
        """
        b_std = self.b_omega_std[0]
        for d in range(1, self.D):
            b_std = b_std * self.b_omega_std[d]
        mask = torch.abs(b_std) < eps_tiny
        b_std = b_std.masked_fill(mask, 0.0)
        return b_std

    def get_W_std_mask(self):
        W_std = self.get_W_std()
        return W_std == 0


    def get_b_std_mask(self):
        b_std = self.get_b_std()
        return b_std == 0