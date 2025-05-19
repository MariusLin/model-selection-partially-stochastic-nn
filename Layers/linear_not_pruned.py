import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from Layers.partially_stochastic_linear_not_pruned import PartiallyStochasticLinearNP

"""
Implements a linear layer with fixed parameters
"""
class LinearNP(nn.Module):
    def __init__(self, in_features, out_features, weight = None, bias = None, scaled_variance = False):
        super(LinearNP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaled_variance = scaled_variance
        if weight is None:
            # Initialize w_j \sim \mathcall{N}(0, 1/n_{in})
            self.weights = nn.Parameter(torch.randn(in_features, out_features) * (1. / in_features)**0.5, requires_grad= True)
        else:
            self.weights = nn.Parameter(weight.data, requires_grad= True)
        if bias is None:
            # see Kolb et al. 2025
            self.bias = nn.Parameter(torch.ones(out_features), requires_grad= True)
        else:
            self.bias = nn.Parameter(bias.data, requires_grad= True)

    def __call__(self, x):
        return self.forward(x)

    def reset_parameters(self):
        std = 1.
        if not self.scaled_variance:
            std = std / math.sqrt(self.in_features)
        init.normal_(self.weight, 0, std)
        init.constant_(self.bias, 1)

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.weights
        if self.scaled_variance:
            W = W / math.sqrt(self.in_features)
        b = self.bias
        return torch.matmul(X, W) + b
    
    def sample_predict(self, X, n_samples):
        """
        Since this layer is determinsitic performing a forward pass is equivalent to performing sample prediction
        """
        X.float()
        return self.forward(X)
    

    def convert_to_partially_stochastic(self, pruned, pos_stochasticity, num_stochastic):
        b = self.bias.data.clone().detach()
        w = self.weights.data.clone().detach()
        init_std = torch.std(w) # Initial standard deviation as of all weights
        return PartiallyStochasticLinearNP(in_features = self.in_features, out_features=self.out_features, 
                                         weight=w, bias=b, pruned=pruned, init_std=init_std,pos_stochasticity=pos_stochasticity, 
                                         num_stochastic=num_stochastic, scaled_variance=self.scaled_variance)
