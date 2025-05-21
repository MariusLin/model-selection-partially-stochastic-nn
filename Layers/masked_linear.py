import math
import torch
import torch.nn as nn


class MaskedLinear(nn.Module):
    def __init__(self, n_in, n_out, b_det_mask, W_det_mask, det_training, scaled_variance=True, device = "cpu"):
        """Initialization.

        Args:
            n_in: int, the size of the input data.
            n_out: int, the size of the output.
        """
        super(MaskedLinear, self).__init__()
        self.device = device
        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.W_det_mask = W_det_mask.to(self.device)
        self.b_det_mask = b_det_mask.top(self.device)
        self.det_training = det_training
        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_det_mask)  
        self.register_buffer("bias_mask", self.b_det_mask) 

        if det_training:
            grad_mask_weight = self.W_det_mask
            grad_mask_bias = self.b_det_mask
        else:
            grad_mask_weight = ~self.W_det_mask
            grad_mask_bias = ~self.b_det_mask

        self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.n_out), True).to(self.device)
        self.weight_hook = self.W.register_hook(lambda grad: grad * grad_mask_weight)
        self.bias_hook = self.b.register_hook(lambda grad: grad * grad_mask_bias)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # Reinitialize only the weights where the mask is 1
            std = 1.
            if not self.scaled_variance:
                std = std / math.sqrt(self.n_in)

        # Sample new weights/biases
        new_W = torch.normal(mean=0.0, std=std, size=self.W.shape, dtype=self.W.dtype).to(self.device)
        new_b = torch.zeros_like(self.b).to(self.device)

        # Ensure masks are on the same device
        self.W_det_mask = self.W_det_mask.to(self.device)
        self.b_det_mask = self.b_det_mask.to(self.device)

        if self.det_training:
            # Apply new values only where the mask is 1
            self.W.data = self.W.data * (~self.W_det_mask) + new_W * self.W_det_mask
            self.b.data = self.b.data * (~self.b_det_mask) + new_b * self.b_det_mask
        else:
            # Apply new values only where the mask is 1
            self.W.data = new_W * (~self.W_det_mask) + self.W.data * self.W_det_mask
            self.b.data = new_b * (~self.b_det_mask) + self.b.data * self.b_det_mask

    def forward(self, X):
        """Performs forward pass given input data.

        Args:
            X: torch.tensor, [batch_size, input_dim], the input data.

        Returns:
            output: torch.tensor, [batch_size, output_dim], the output data.
        """
        W = self.W
        if self.scaled_variance:
            W = W / math.sqrt(self.n_in)
        b = self.b
        return torch.mm(X, W) + b
    
    
    def sample_predict(self, X, n_samples):
        """Makes predictions using a set of sampled weights.

        Args:
            X: torch.tensor, [n_samples, batch_size, input_dim], the input
                data.
            n_samples: int, the number of weight samples used to make
                predictions.

        Returns:
            torch.tensor, [n_samples, batch_size, output_dim], the output data.
        """
        X = X.float()
        Ws = self.W.repeat(n_samples, 1, 1)
        bs = self.b.repeat(n_samples, 1, 1)

        return torch.matmul(X, Ws) + bs

    def change_hook(self, det_training, dropout, p):
        self.weight_hook.remove()
        self.bias_hook.remove()
        if det_training:
            if dropout:
                grad_mask_weight = self.W_det_mask * torch.bernoulli(torch.full(self.W_det_mask.shape, p)).to(self.device)
                grad_mask_bias = self.b_det_mask * torch.bernoulli(torch.full(self.b_det_mask.shape, p)).to(self.device)
            else:
                grad_mask_weight = self.W_det_mask
                grad_mask_bias = self.b_det_mask
        else:
            grad_mask_weight = ~self.W_det_mask
            grad_mask_bias = ~self.b_det_mask
        
        self.weight_hook = self.W.register_hook(lambda grad: grad * grad_mask_weight)
        self.bias_hook = self.b.register_hook(lambda grad: grad * grad_mask_bias)
        self.det_training = det_training

    def change_hook_wcp(self):
        weight_eta = torch.abs(self.W.grad)
        bias_eta = torch.abs(self.b.grad)
        # Remove hooks
        self.weight_hook.remove()
        self.bias_hook.remove()
        # Get new masks
        grad_mask_weight = self.W_det_mask * self._sample_tensor_from_wcp(self.W.shape, weight_eta)
        grad_mask_bias = self.b_det_mask * self._sample_tensor_from_wcp(self.b.shape, bias_eta)
        # Get new hooks
        self.weight_hook = self.W.register_hook(lambda grad: grad * grad_mask_weight)
        self.bias_hook = self.b.register_hook(lambda grad: grad * grad_mask_bias)

    # Sample a single tensor from Normal(m, s)
    def _sample_tensor_from_wcp(self, shape, eta):
        eta = eta.to(self.device)

        # Output tensor initialized to zeros
        out = torch.zeros(shape, device=self.device)

        # Mask where eta is not zero
        nonzero_mask = eta != 0
        if not nonzero_mask.any():
            return out

        eta_nonzero = eta[nonzero_mask]
        num_samples = eta_nonzero.numel()

        def sample_m_s(num_samples, eta_vals, M=10.0, S=10.0, epsilon=1e-3):
            def wcp_density(m, s, eta_vals):
                r = torch.sqrt(m**2 + s**2)
                return eta_vals * torch.exp(-eta_vals * r) / (math.pi * r)

            samples_m = torch.empty(num_samples, device=self.device)
            samples_s = torch.empty(num_samples, device=self.device)

            batch_size = 4096
            collected = 0

            while collected < num_samples:
                remaining = num_samples - collected

                batch_eta = eta_vals[collected:collected + remaining]
                current_batch_size = batch_eta.shape[0]

                m = (torch.rand(batch_size, device=self.device) * 2 - 1) * M
                s = torch.rand(batch_size, device=self.device) * (S - epsilon) + epsilon

                # Expand eta to match batch for vectorized comparison
                eta_expanded = batch_eta.view(-1, 1)
                m_expanded = m.unsqueeze(0).expand(current_batch_size, -1)
                s_expanded = s.unsqueeze(0).expand(current_batch_size, -1)

                p = wcp_density(m_expanded, s_expanded, eta_expanded)
                max_density = wcp_density(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(epsilon, device=self.device),
                    eta_expanded
                )

                u = torch.rand_like(p) * max_density
                accepted = u < p

                # For each eta value, accept the first valid sample
                for i in range(current_batch_size):
                    accepted_idx = torch.nonzero(accepted[i], as_tuple=False)
                    if accepted_idx.numel() > 0:
                        idx = accepted_idx[0, 0]
                        samples_m[collected] = m[idx]
                        samples_s[collected] = s[idx]
                        collected += 1
                        if collected == num_samples:
                            break

            return samples_m, samples_s

        m, s = sample_m_s(num_samples, eta_nonzero)
        eps = torch.randn(num_samples, device=self.device)
        samples = m + s * eps

        out[nonzero_mask] = samples
        return out