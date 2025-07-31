import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

"""
This is a slight adaptation of the 2D Convolutional layer taken from Tran et al. 2022
Here, the parameters that are to be treated stochastically are getting the suffix stoch
"""
class StochConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, scaled_variance=True):
        super(StochConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.scaled_variance = scaled_variance

        W_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        b_shape = (self.out_channels)

        self.W_stoch = nn.Parameter(torch.zeros(W_shape), requires_grad=True)
        if self.bias:
            self.b_stoch = nn.Parameter(torch.zeros(b_shape), requires_grad=True)
        else:
            self.register_buffer("b_stoch", torch.zeros(b_shape))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if not self.scaled_variance:
            std = std / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        init.normal_(self.W_stoch, 0, std)
        if self.bias:
            init.constant_(self.b_stoch, 0)

    def forward(self, X):
        W = self.W_stoch
        if self.scaled_variance:
            W = W / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias:
            b = self.b_stoch
        else:
            b = torch.zeros((self.out_channels), device=self.W_stoch.device)
        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)