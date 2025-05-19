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

        self.n_in = n_in
        self.n_out = n_out
        self.scaled_variance = scaled_variance
        self.W_det_mask = W_det_mask
        self.b_det_mask = b_det_mask
        self.det_training = det_training
        self.device = device
        # Register the masks and perform masking
        self.register_buffer("weight_mask", self.W_det_mask)  
        self.register_buffer("bias_mask", self.b_det_mask) 

        if det_training:
            grad_mask_weight = self.W_det_mask
            grad_mask_bias = self.b_det_mask
        else:
            grad_mask_weight = ~self.W_det_mask
            grad_mask_bias = ~self.b_det_mask

        self.W = nn.Parameter(torch.zeros(self.n_in, self.n_out), True)
        self.b = nn.Parameter(torch.zeros(self.n_out), True)
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
            new_W = torch.normal(mean=0.0, std=std, size=self.W.shape, device=self.device, dtype=self.W.dtype)
            new_b = torch.zeros_like(self.b)  # Since init.constant_(b, 0)
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

    def change_hook(self, det_training):
        self.weight_hook.remove()
        self.bias_hook.remove()
        if det_training:
            grad_mask_weight = self.W_det_mask
            grad_mask_bias = self.b_det_mask
        else:
            grad_mask_weight = ~self.W_det_mask
            grad_mask_bias = ~self.b_det_mask
        
        self.weight_hook = self.W.register_hook(lambda grad: grad * grad_mask_weight)
        self.bias_hook = self.b.register_hook(lambda grad: grad * grad_mask_bias)
        self.det_training = det_training