import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utilities.activation_functions import *
from Layers.partially_stochastic_linear_not_pruned import PartiallyStochasticLinearNP

"""
Initializes a partially stochastic multi-layer perceptron
"""

class PSMLP(nn.Module):
    def __init__(self, input_dim, layers, output_layer, norm_layers, activation_fn, 
                 task = "regression", sn2 = 0.1, norm_layer = None):
        super(PSMLP, self).__init__()

        self.norm_layers = norm_layers
        self.layers = layers
        self.output_layer = output_layer

        self.input_dim = input_dim
        self.task = task
        self.sn2 = sn2
        self.norm_layer = norm_layer

        # Setup activation function
        options = {'cos': torch.cos, 'tanh': torch.tanh, 'relu': F.relu,
                   'softplus': F.softplus, 'rbf': rbf, 'linear': linear,
                   'sin': torch.sin, 'leaky_relu': F.leaky_relu, 
                   'swish': swish}
        if activation_fn in options:
            self.activation_fn = options[activation_fn]
        else:
            self.activation_fn = activation_fn


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
            X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, self.input_dim)
        X = X.to(next(self.parameters()).device)
        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)

        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance (layer, PartiallyStochasticLinearNP):
                layer.reset_parameters()
    
    def net_predict(self, X, weights, log_softmax=False):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            log_softmax: bool, indicates whether or not return the log softmax
                values.

        Returns:
            torch.tensor, [batch_size, output_dim], the output data.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.view(-1, self.input_dim)
        X = X.to(next(self.parameters()).device)
        for linear_layer, norm_layer in zip(list(self.layers),
                                            list(self.norm_layers)):
            if isinstance(linear_layer, PartiallyStochasticLinearNP):    
                X = self.activation_fn(norm_layer(linear_layer.predict(X, weights)))
            else:
                X = self.activation_fn(norm_layer(linear_layer(X)))

        X = self.output_layer(X)

        if (self.task == "classification") and log_softmax:
            X = F.log_softmax(X, dim=1)

        return X.detach().cpu().numpy()
    
    def predict(self, x_test, sampled_weights, return_individual_predictions=False,
                return_raw_predictions=False):
        """Predicts mean and variance for the given test point.

        Args:
            x_test: numpy array, the test datapoint.
            return_individual_predictions: bool, if True also the predictions
                of the individual models are returned.
            return_raw_predictions: bool, indicates whether or not return
                the raw predictions along with the unnormalized predictions.

        Returns:
            a tuple consisting of mean and variance.
        """
        # Make predictions for each sampled weights
        predictions = np.array([
            self.net_predict(x_test, weights=weights)
            for weights in sampled_weights])
        # Calculates the predictive mean and variance
        pred_mean = np.array(predictions.mean(axis=0))
        pred_var = np.array(predictions.std(axis=0)**2 + self.sn2)

        if return_raw_predictions:
            raw_predictions = copy.deepcopy(predictions).squeeze()


        if return_individual_predictions:
            if return_raw_predictions:
                return pred_mean, pred_var, predictions, raw_predictions
            else:
                return pred_mean, pred_var, predictions

        return pred_mean, pred_var


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
