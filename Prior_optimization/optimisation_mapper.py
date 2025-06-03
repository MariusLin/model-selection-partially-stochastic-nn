import numpy as np
import torch
from Utilities.util import ensure_dir
import os
import torch.profiler


class PriorOptimisationMapper():
    def __init__(self, out_dir, kernel, device = "cpu"):
        self.out_dir = out_dir
        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)
        self.device = device
        self.kernel = kernel

    def sigmoid_schedule(self, epoch, n_epochs, shift, scale, tau = 0.7, start = 0, end = 3):
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        v_start = sigmoid(start / tau)
        v_end = sigmoid(end / tau)
        return scale * ((v_end - sigmoid ((epoch/n_epochs * (end-start) + start)/tau)) / (v_end - v_start)) + shift
        

    def to(self, device):
        self.device = torch.device(device)
        return self 

    def sample_covariance_difference(self, nnet_samples, kernel_matrix):
        """
        Computes the covariance between B samples (each with D features),
        compares it with a kernel matrix, and returns the sum of squared differences.

        Args:
            nnet_samples: Tensor of shape [B, N, D]
            kernel_matrix: Tensor of shape [B, B]

        Returns:
            Scalar tensor: sum of squared differences between covariance and kernel
        """
        B, N, D = nnet_samples.shape  # [B, N, D]
        # Mean across the N samples for each B
        data = nnet_samples.mean(dim=1)  # [B, D]
        # Center across feature dimension
        data_centered = data - data.mean(dim=0, keepdim=True)  # [B, D]
        # Handle case where D == 1 separately to avoid division by 0
        if D == 1:
            cov = data_centered @ data_centered.T  # [B, B], no division
        else:
            cov = data_centered @ data_centered.T / (D - 1)

        # Compute squared Frobenius norm of difference
        diff = cov - kernel_matrix
        # Zero out diagonal elements
        diff.fill_diagonal_(0)
        return (diff**2).sum()
    
    def variance_calculator(self, nnet_samples):
        # nnet_samples: shape [B, N, D]
        var_over_samples = torch.var(nnet_samples, dim=1, unbiased=False)  # shape [B, D]

        total_variance = var_over_samples.sum()  # scalar

        return 1/(total_variance + 1.19e-7)

    def compute_kernel_matrix(self, X):
        """
        Compute the [B, B] kernel matrix from [B, D] input samples using the given kernel function.

        Args:
            X: Tensor of shape [B, D]
            kernel_fn: function taking two [D]-shaped tensors and returning a scalar

        Returns:
            Tensor of shape [B, B] — the kernel matrix
        """
        X = X.reshape((-1, self.kernel.input_dim))
        return self.kernel.K(X,X)

    def optimize(self, net, data_generator, n_data, num_iters, output_dim, lambd, shift = 1, scale = 5,
                n_samples=128, lr=1e-1, print_every=100, save_ckpt_every = 500):
        net = torch.compile(net)
        sdist_hist = np.array([])
        pruned_hist = np.array([])
        prior_optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
        # Cosine Annealing worked too
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prior_optimizer, T_max=num_iters)
        omega_params = []
        for layer in net.layers:
            layer_omega = []
            for name, param in layer.named_parameters():
                if "omega" in name:
                    layer_omega.append(param.view(-1)) 
            if layer_omega:
                layer_omega_row = torch.cat(layer_omega)
                omega_params.append(layer_omega_row)
        
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        prof.start()
        # Prior loop
        for it in range(1, num_iters+1): 
            # Get regularization parameters for loss
            reg_loss = self.sigmoid_schedule(it, num_iters, shift = shift, scale = scale)
            # Draw X
            X = data_generator.get(n_data)
            X = X.to(self.device)
            # Calculate Kernel Matrix
            kernel_matrix = self.compute_kernel_matrix(X)
            # Draw functions from Neural Network
            nnet_samples = net.sample_functions(
                X, n_samples).float().to(self.device)
            if output_dim > 1:
                nnet_samples = nnet_samples.squeeze()
            prior_optimizer.zero_grad()
            # Stay on CPU
            reg_loss_vector = torch.tensor([
                torch.sum((omega_params_per_layer.cpu()) ** 2).item()
                for omega_params_per_layer in omega_params
            ], device='cpu')  # directly create tensor on CPU

            # Now do matmul with lambd (assumed to be on CPU or tiny)
            regularization = torch.matmul(lambd.cpu(), reg_loss_vector)

            # Move final result to GPU only if needed
            regularization = regularization.to(self.device)
            # reg_loss_vector = torch.tensor([torch.sum(omega_params_per_layer ** 2) \
            #                    for omega_params_per_layer in omega_params]).to(self.device)
            # regularization = torch.matmul(lambd, reg_loss_vector)
                # Compute the distance to the kernel matrix
            sdist = self.sample_covariance_difference(nnet_samples, kernel_matrix) + \
                    self.variance_calculator(nnet_samples)
            loss = reg_loss * sdist+regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(omega_params, 100.)
            prior_optimizer.step()
            scheduler.step()

            num_pruned = 0
            for layer in net.layers:
                num_pruned += (layer.get_W_std() == 0).sum().item()
                num_pruned += (layer.get_b_std() == 0).sum().item()
            pruned_hist = np.append(pruned_hist, num_pruned)

            sdist_hist = np.append(sdist_hist, float(sdist))
            if (it % print_every == 0) or it == 1:
                print(">>> Iteration # {:3d}: "
                          "Difference from GP {:.4f}".format(
                            it, float(sdist)), f"Number of pruned stochastic weights: {num_pruned}")
            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                checkpoint = {}
                for i, layer in enumerate(net.layers):
                    W_std = layer.get_W_std()
                    b_std = layer.get_b_std()
                    checkpoint[f"layer_{i}_W_std"] = W_std
                    checkpoint[f"layer_{i}_b_std"] = b_std
                torch.save(checkpoint, path)
            if it <= 1000:
                prof.step()
            if it == 1001:
                prof.stop()
                # Save profiler summary to a text file instead of printing
                with open("profiler_summary_optim.txt", "w") as f:
                    f.write(prof.key_averages().table())
        return pruned_hist, sdist_hist