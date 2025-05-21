import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from Utilities.activation_functions import *
from Layers.linear_not_pruned import LinearNP
from Networks.partially_stochastic_mlp import PSMLP

"""
Initializes a plain multi-layer perceptron
"""

def init_norm_layer(input_dim, norm_layer):
    if norm_layer == "batchnorm":
        return nn.BatchNorm1d(input_dim, eps=0, momentum=None,
                              affine=False, track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()

class DetMLPNP(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dims, activation_fn, 
                 norm_layer = None, task = "regression", scaled_variance = True):
        super(DetMLPNP, self).__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.norm_layer = norm_layer
        self.task = task

        # Setup activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': torch.sin, 'leaky_relu': F.leaky_relu, 
                   'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn

        self.layers = nn.ModuleList([LinearNP(
            input_dim, hidden_dims[0], scaled_variance=scaled_variance)])
        self.norm_layers = nn.ModuleList([init_norm_layer(
            hidden_dims[0], self.norm_layer)])
        for i in range(1, len(hidden_dims)):
            self.layers.add_module(
                "linear_{}".format(i), LinearNP(hidden_dims[i-1], hidden_dims[i],
                scaled_variance=scaled_variance))
            self.norm_layers.add_module(
                "norm_{}".format(i), init_norm_layer(hidden_dims[i],
                                                     self.norm_layer))
        self.output_layer = LinearNP(hidden_dims[-1], output_dim,
                                   scaled_variance=scaled_variance)
        

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, LinearNP):
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
        X = X.view(-1, self.input_dim)

        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)

        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X
    
    def convert_to_partially_stochastic_mlp(self,pruned, pos_stochasticity, num_stochastic, stoch_layer_inds):
        # Copy all layers and norm layers
        layers = nn.ModuleList([copy.deepcopy(layer) for layer in self.layers])

        for ind, layer in enumerate(self.layers):
            if ind in stoch_layer_inds:
                layers[ind] = layers[ind].convert_to_partially_stochastic(pruned, pos_stochasticity, num_stochastic)
            else:
               layers[ind].bias.requires_grad_(False)
               layers[ind].weights.requires_grad_(False)
        if pruned:
            for ind, layer in enumerate(layers):
                if num_stochastic <= 0:
                    break
                num_pruned = torch.sum(layer.weights == 0).item()
                if num_pruned>0:
                    layers[ind] = layers[ind].convert_to_partially_stochastic(pruned, pos_stochasticity, min(num_pruned, num_stochastic))
                num_stochastic -= num_pruned
        output_layer = copy.deepcopy(self.output_layer)
        output_layer.bias.requires_grad_(False)
        output_layer.weights.requires_grad_(False)
                    

        return PSMLP(self.input_dim, layers, output_layer, self.norm_layers, self.activation_fn, self.task, norm_layer = self.norm_layer)
    
    def convert_to_partially_stochastic_mlp_all_params(self,num_stochastic, stoch_layer_inds):
        # Copy all layers and norm layers
        layers = nn.ModuleList([copy.deepcopy(layer) for layer in self.layers])
        # Make layers partially stochastic
        for ind in range(len(layers)):
            if ind in stoch_layer_inds:
                layers[ind] = layers[ind].convert_to_partially_stochastic_all_params(num_stochastic)
        output_layer = copy.deepcopy(self.output_layer)
        return PSMLP(self.input_dim, layers, output_layer, self.norm_layers, self.activation_fn, self.task, norm_layer = self.norm_layer)