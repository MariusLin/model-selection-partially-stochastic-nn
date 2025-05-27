"""
Implements a linear layer that has both stochastic and deterministic weights
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartiallyStochasticLinearNP(nn.Module):
    def __init__(self, in_features, out_features, weight, bias, pruned, pos_stochasticity, 
                 num_stochastic, init_std = 1.0, scaled_variance=False):
        super(PartiallyStochasticLinearNP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaled_variance = scaled_variance

        # Save full weight matrix and bias
        self.register_buffer("weights", weight.clone())
        self.register_buffer("bias", bias.data.clone().detach())
        # Select stochastic indices
        self.stochastic_indices = []
        count = 0
        if pruned:
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    if self.weights[i, j] == 0 and count < num_stochastic:
                        self.stochastic_indices.append((i, j))
                        count += 1
        else:
            if pos_stochasticity == "highest":
                # Flatten weights and select top-k absolute values
                flat_weight = self.weights.view(-1)
                _, flat_indices = torch.topk(torch.abs(flat_weight), k=num_stochastic)
                for idx in flat_indices:
                    i, j = divmod(idx.item(), self.in_features)
                    self.stochastic_indices.append((j,i))
            elif pos_stochasticity == "lowest":
                # Flatten weights and select top-k absolute values
                flat_weight = self.weights.view(-1)
                _, flat_indices = torch.topk(torch.abs(flat_weight), k=num_stochastic, largest=False)
                for idx in flat_indices:
                    i, j = divmod(idx.item(), self.in_features)
                    self.stochastic_indices.append((j,i))
            else:
                torch.seed()
                # Pick random indices
                flat_weight = self.weights.view(-1)
                num_weights = flat_weight.numel()

                # Randomly choose indices without replacement
                rand_indices = torch.randperm(num_weights)[:num_stochastic]

                # Map flat indices to 2D indices (i, j)
                for idx in rand_indices:
                    i, j = divmod(idx.item(), self.in_features)
                    self.stochastic_indices.append((j,i))

        # Parameters for stochastic weights
        self.initial_mu = nn.Parameter(torch.tensor(
            [self.weights[i, j] for (i, j) in self.stochastic_indices], dtype=torch.float32), requires_grad=False)
        self.initial_std = nn.Parameter(torch.full_like(self.initial_mu, fill_value=init_std, dtype=torch.float32),
                                        requires_grad=False)
        self.stochastic_mu = nn.Parameter(self.initial_mu, requires_grad=True)
        self.stochastic_std = nn.Parameter(self.initial_std, requires_grad=True)
        self.initial_samples = self.stochastic_mu + torch.rand_like(self.stochastic_mu) * self.stochastic_std
        self.stochastic_weights = self.initial_samples
        self.weights_test = self.weights.clone()
        for sampled_value, (i, j) in zip(self.stochastic_weights, self.stochastic_indices):
            self.weights_test[i, j] = sampled_value
        

    def forward(self, input):
            # Ensure all tensors are on the same device as the input
            device = input.device
            weights = self.weights.to(device)
            bias = self.bias.to(device)
            stochastic_weights = self.stochastic_weights.to(device)

            # Start with a copy of the deterministic weights
            weights = weights.clone()
            # Sample stochastic values
            sampled_values = stochastic_weights
            # Insert sampled stochastic weights into the weight matrix
            for sampled_value, (i, j) in zip(sampled_values, self.stochastic_indices):
                weights[i, j] = sampled_value
            if self.scaled_variance:
                weights = weights / math.sqrt(self.in_features)
            return torch.matmul(input, weights) + bias

    def predict(self, input, sampled_weights):
        # Ensure all tensors are on the same device as the input
        device = input.device
        weights = self.weights.to(device)
        bias = self.bias.to(device)
        sampled_weights = sampled_weights.to(device)

        weights = weights.clone()
        for sampled_weight, (i, j) in zip(sampled_weights, self.stochastic_indices):
            weights[i, j] = sampled_weight
        if self.scaled_variance:
            weights = weights / math.sqrt(self.in_features)
        return torch.matmul(input, weights) + bias

    
    def sample_predict(self, X, n_samples):
        """Makes predictions using a set of sampled weights.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.
            n_samples: int, the number of weight samples used to make predictions.

        Returns:
            torch.tensor, [n_samples, batch_size, output_dim], the output data.
        """
        X = X.float()
        device = X.device

        # Create n_samples copies of the base weights
        Ws = self.weights.unsqueeze(0).repeat(n_samples, 1, 1).to(device)
        # Sample different stochastic weights for each sample
        sampled_weights = self.stochastic_mu.to(device) + torch.rand(
            (n_samples, *self.stochastic_mu.shape),
            device=device
        ) * self.stochastic_std.to(device)
        # Replace the entries at stochastic_indices for each sample
        for i in range(n_samples):
            for ind, index in enumerate(self.stochastic_indices):
                Ws[i][index] = sampled_weights[i][ind]

        # Optionally scale weights if required
        if self.scaled_variance:
            Ws = Ws / math.sqrt(self.in_features)

        # Expand bias for each sample
        bs = self.bias.unsqueeze(0).repeat(n_samples, 1, 1).to(device)
        return torch.matmul(X, Ws) + bs

    def reset_parameters(self):
        if not self.scaled_variance:
            self.initial_std /= math.sqrt(self.in_features)
        with torch.no_grad():
            self.stochastic_mu.data.copy_(self.initial_mu)
            self.stochastic_std.data.copy_(self.initial_std)