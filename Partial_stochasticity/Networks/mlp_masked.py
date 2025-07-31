import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utilities.activation_functions import *
from Partial_stochasticity.Layers.masked_linear import MaskedLinear
from Partial_stochasticity.Layers.factorized_linear import FactorizedLinear

"""
This is a normal MLP with masked layers to split deterministic and stochastic weights
True means deterministic and false means stochastic
"""
def init_norm_layer(input_dim, norm_layer):
    if norm_layer == "batchnorm":
        return nn.BatchNorm1d(input_dim, eps=0, momentum=None,
                              affine=False, track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()

class MLPMasked(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation_fn, weight_masks, bias_masks, D, 
                 prior_W_std_list = [],  prior_b_std_list = [], W_std = None, b_std = None, W_std_out = None, 
                 b_std_out = None, scaled_variance=True, norm_layer=None, task="regression", device = "cpu"):
        super(MLPMasked, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.norm_layer = norm_layer
        self.task = task
        self.weight_masks = weight_masks
        self.bias_masks = bias_masks
        self.device = device
        self.prior_W_std_list = prior_W_std_list
        self.prior_b_std_list = prior_b_std_list

        # Setup activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': torch.sin, 'leaky_relu': F.leaky_relu, 
                   'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        self.layers = nn.ModuleList([MaskedLinear(
            input_dim, hidden_dims[0], self.bias_masks[0], self.weight_masks[0], D = D, W_std = W_std, b_std = b_std, 
            prior_b_std=self.prior_b_std_list[0], prior_W_std=self.prior_W_std_list[0] , scaled_variance=scaled_variance, device = self.device)])
        self.norm_layers = nn.ModuleList([init_norm_layer(
            hidden_dims[0], self.norm_layer)])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                "linear_{}".format(i), MaskedLinear(hidden_dims[i-1], hidden_dims[i], self.bias_masks[i],
                self.weight_masks[i], D = D, W_std = W_std, b_std = b_std, scaled_variance=scaled_variance, 
                prior_b_std=self.prior_b_std_list[i], prior_W_std=self.prior_W_std_list[i], device=self.device))
            self.norm_layers.add_module(
                "norm_{}".format(i), init_norm_layer(hidden_dims[i],
                                                     self.norm_layer))
        self.output_layer = FactorizedLinear(hidden_dims[-1], output_dim, D = D, W_std = W_std_out, 
                        b_std = b_std_out, scaled_variance=scaled_variance, device=self.device)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, MaskedLinear):
                m.reset_parameters()

    def forward(self, X, log_softmax=False):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            log_softmax: bool, indicates whether or not return the log softmax
                values.

        Returns:
            torch.tensor, [batch_size, output_dim], the output data.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        X = X.view(-1, self.input_dim)

        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)

        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X

    def predict(self, X):
        """Make predictions given input data.

        Args:
            x: torch tensor, shape [batch_size, input_dim]

        Returns:
            torch tensor, shape [batch_size, num_classes], the predicted
                probabilites for each class.
        """
        self.eval()
        if self.task == "classification":
            return torch.exp(self.forward(X, log_softmax=True))
        else:
            return self.forward(X, log_softmax=False)
            

    def sample_functions(self, X, n_samples):
        """Performs predictions using `n_samples` set of weights.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [batch_size, n_samples, output_dim], the output
            data.
        """
        X = X.view(-1, self.input_dim)
        X = torch.unsqueeze(X, 0).repeat([n_samples, 1, 1])
        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            if self.norm_layer is None:
                X = self.activation_fn(linear_layer.sample_predict(X, n_samples))
            else:
                X = linear_layer.sample_predict(X, n_samples)
                out = torch.zeros_like(X, device=X.device, dtype=X.dtype)
                for i in range(n_samples):
                    out[i, :, :] = norm_layer(X[i, :, :])
                X = self.activation_fn(out)

        X = self.output_layer.sample_predict(X, n_samples)
        X = torch.transpose(X, 0, 1)

        return X
    
    def get_nums_pruned(self):
        """
        This iterates through all layers and determines the number of pruned weights
        """
        num_pruned_stoch = 0
        num_pruned_det = 0
        for layer in self.layers + [self.output_layer]:
            nums_pruned_W = layer.get_nums_pruned_W()
            nums_pruned_b = layer.get_nums_pruned_b()
            num_pruned_stoch += nums_pruned_W[0]
            num_pruned_stoch += nums_pruned_b[0]
            num_pruned_det += nums_pruned_W[1]
            num_pruned_det += nums_pruned_b[1]
        return num_pruned_stoch, num_pruned_det