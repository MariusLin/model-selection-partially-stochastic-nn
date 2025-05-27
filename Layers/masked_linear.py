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
        self.b_det_mask = b_det_mask.to(self.device)
        self.det_training = det_training
        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_det_mask)  
        self.register_buffer("bias_mask", self.b_det_mask) 

        if det_training:
            self.grad_mask_weight = self.W_det_mask
            self.grad_mask_bias = self.b_det_mask
        else:
            self.grad_mask_weight = ~self.W_det_mask
            self.grad_mask_bias = ~self.b_det_mask

        self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.n_out), True).to(self.device)
        self.weight_hook = self.W.register_hook(lambda grad: grad * self.grad_mask_weight)
        self.bias_hook = self.b.register_hook(lambda grad: grad * self.grad_mask_bias)
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

    def change_hook_wcp(self, step, num_steps):
            weight_grad_shape = self.W.grad.shape
            bias_grad_shape = self.b.grad.shape
            # Remove hooks
            self.weight_hook.remove()
            self.bias_hook.remove()
            # Get new masks
            grad_mask_weight = self.W_det_mask * self.perform_wcp_regularization(weight_grad_shape, step, num_steps)
            grad_mask_bias = self.b_det_mask * self.perform_wcp_regularization(bias_grad_shape, step, num_steps)
            # Get new hooks
            self.weight_hook = self.W.register_hook(lambda grad: grad * grad_mask_weight)
            self.bias_hook = self.b.register_hook(lambda grad: grad * grad_mask_bias)

    def perform_wcp_regularization(self, eta_shape, step, num_steps):
        def zero_one_schedule(i, n_epochs):
            t = (1-1e-8*n_epochs)/(1- n_epochs)
            m = 1e-8-t
            return m*i +t
            
        # Here perform sampling
        def sample_from_wcp(eta_vals, max_uniform):
            new_mean = 1- max_uniform
            eta_safe = eta_vals.clamp(min=1e-6)  # prevent invalid exponential rates
            r = torch.distributions.Exponential(eta_safe).sample()
            # s>=0
            t = torch.pi * torch.rand(eta_vals.shape, device=eta_vals.device)
            m = r * torch.cos(t) + new_mean
            s = r * torch.sin(t)
            return m, s
        max_uniform = zero_one_schedule(step, num_steps)
        eta_vals = torch.rand(*eta_shape) * max_uniform
        eta_vals = eta_vals.to(self.device)

        m, s = sample_from_wcp(eta_vals, max_uniform)
        sampled_values = torch.normal(mean=m, std=s)
        return sampled_values