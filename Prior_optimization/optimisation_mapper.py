import numpy as np
import torch
from Utilities.util import ensure_dir
import os


class PriorOptimisationMapper():
    def __init__(self, out_dir, device = "cpu"):
        self.out_dir = out_dir
        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)
        self.device = device


    def to(self, device):
        self.device = torch.device(device)
        return self 


    def energy_distance(self, x, y):
        """
        Computes the energy distance between two batches of samples x and y.
        x: Tensor of shape (n, d)
        y: Tensor of shape (m, d)
        Returns a scalar tensor.
        """
        def pairwise_distances(a, b):
            return torch.cdist(a, b, p=2)

        d_xy = pairwise_distances(x, y).mean()
        d_xx = pairwise_distances(x, x).mean()
        d_yy = pairwise_distances(y, y).mean()

        return 2 * d_xy - d_xx - d_yy

    def optimize(self, net, gp, data_generator, n_data, num_iters, output_dim, D, lambd,
                n_samples=128, lr=1e-1, print_every=100, gpu_gp = True, save_ckpt_every = 500):
        
        sdist_hist = np.array([])
        pruned_hist = np.array([])

        prior_optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
        
        omega_params = []
        for layer in net.layers:
            layer_omega = []
            for name, param in layer.named_parameters():
                if "omega" in name:
                    layer_omega.append(param.view(-1)) 
            if layer_omega:
                layer_omega_row = torch.cat(layer_omega)
                omega_params.append(layer_omega_row)

        # Prior loop
        for it in range(1, num_iters+1):  
            # Draw X
            X = data_generator.get(n_data)
            X = X.to(self.device)
            if not gpu_gp:
                X = X.to("cpu")

            # Draw functions from Gaussian Process
            gp_samples = gp.sample_functions(
                        X.double(), n_samples).detach().float().to(self.device)
            if output_dim > 1:
                gp_samples = gp_samples.squeeze()

            if not gpu_gp:
                X = X.to(self.device)

            # Draw functions from Neural Network
            nnet_samples = net.sample_functions(
                        X, n_samples).float().to(self.device)
            if output_dim > 1:
                nnet_samples = nnet_samples.squeeze()

            prior_optimizer.zero_grad()
            reg_loss_vector = (1 / D) * torch.tensor([torch.sum(omega_params_per_layer ** 2) \
                                                        for omega_params_per_layer in omega_params]).to(self.device)
            regularization = torch.matmul(lambd, reg_loss_vector)
            # Compute the energy distance
            sdist = self.energy_distance(nnet_samples, gp_samples)
            loss = sdist+regularization
            loss.backward()
            prior_optimizer.step()

            num_pruned = 0
            for layer in net.layers:
                num_pruned += (layer.get_W_std() == 0).sum().item()
                num_pruned += (layer.get_b_std() == 0).sum().item()
            pruned_hist = np.append(pruned_hist, num_pruned)

            sdist_hist = np.append(sdist_hist, float(sdist))
            if (it % print_every == 0) or it == 1:
                print(">>> Iteration # {:3d}: "
                                    "Energy Distance {:.4f}".format(
                                        it, float(sdist)), f"Number of pruned stochastic weights: {num_pruned}")
            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                torch.save(net.state_dict(), path)

        return pruned_hist, sdist_hist