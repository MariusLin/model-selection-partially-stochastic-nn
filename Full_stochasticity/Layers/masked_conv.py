import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is a 2D Convolutional Layer 
It has two additional parameters, namely the masks indicating wheter the bias or weight at a certain index is
stochastic or deterministic (1 means deterministic and 0 means stochastic)
"""
class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, 
                  W_det_mask = None, b_det_mask = None, scaled_variance=True, device = "cpu"):
        super(MaskedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.scaled_variance = scaled_variance
        self.device = device
        # Here the masks are transformed into a stochastic mask (1 means stochastic and 0 means deterministic)
        self.W_stoch_mask = ~W_det_mask.to(self.device)
        if b_det_mask:
            self.b_stoch_mask = ~b_det_mask.to(self.device)

        self.W_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        self.b_shape = (self.out_channels)

        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_stoch_mask) 
        if b_det_mask: 
            self.register_buffer("bias_mask", self.b_stoch_mask) 

        full_weight = torch.zeros(self.W_shape, device = self.device)
        # Only the weights that are stochastic are trainable
        self.W_stoch = nn.Parameter(full_weight[self.W_stoch_mask])
        if self.bias:
            full_bias = torch.zeros(self.b_shape, device = self.device)
            # Only the biases that are stochastic are trainable
            self.b_stoch = nn.Parameter(full_bias[self.b_stoch_mask])
        else:
            self.register_buffer("b", torch.zeros(self.b_shape))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.
        if not self.scaled_variance:
            std = std / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        new_W = torch.rand(self.W_shape, device=self.device) * std
        self.W_stoch.data = new_W[self.W_stoch_mask]
        if self.bias:
            new_b = torch.zeros(self.b_shape, device = self.device)
            self.b_stoch.data = new_b[self.b_stoch_mask]

    def forward(self, X):
        X = X.to(self.device)
        W = self.get_W()
        if self.scaled_variance:
            W = W / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        if self.bias:
            b = self.get_b()
        else:
            b = torch.zeros((self.out_channels), device=self.device)
        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)
    
    def get_W(self):
        """
        This function returns a properly shaped W. The deterministic weights are set to 0
        """
        full_W = torch.zeros(self.W_shape, device=self.device)
        full_W[self.W_stoch_mask] = self.W_stoch
        return full_W
    
    def get_b(self):
        """
        This function returns a properly shaped b. The deterministic indexes are set to 0
        """
        full_b = torch.zeros(self.b_shape, device=self.device)
        full_b[self.b_stoch_mask] = self.b_stoch
        return full_b