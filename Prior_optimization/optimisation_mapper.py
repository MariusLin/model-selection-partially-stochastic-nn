import numpy as np
import torch
from Utilities.util import ensure_dir
import os
import torch.nn as nn

"""
This is a mapper to perform the optimization of the prior 
"""
class PriorOptimisationMapper():
    def __init__(self, out_dir, gp, out_det = True, device = "cpu"):
        self.out_dir = out_dir
        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)
        self.device = device
        self.gp = gp
        self.out_det = out_det

    def to(self, device):
        self.device = torch.device(device)
        return self
    
    """
    @inproceedings{NEURIPS2023,
    author = {Nguyen, Khai and Ho, Nhat},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
    pages = {18046--18075},
    publisher = {Curran Associates, Inc.},
    title = {Energy-Based Sliced Wasserstein Distance},
    url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/3a23caeb904c822575fa56fb114ca499-Paper-Conference.pdf},
    volume = {36},
    year = {2023}
    }
    """
    def ebswd(self, nnet_samples, gp_samples, output_dim, L=10, p=2, device="cpu"):
        """
        Vectorized computation of EBSWD

        nnet_samples: tensor of shape [batch_size, num_functions, output_dim]
        gp_samples: tensor of shape [batch_size, num_functions, output_dim]
        """
        dist = 0
        batch_size, *_ = gp_samples.shape

        for dim in range(output_dim):
            # Extract and reshape slices to [num_functions, batch_size]
            nnet_samples_slice = nnet_samples[:, :, dim].permute(1, 0).contiguous()
            gp_samples_slice = gp_samples[:, :, dim].permute(1, 0).contiguous()

            # Generate random directions: shape [L, num_functions]
            theta = torch.randn((L, batch_size), device=device)
            theta = theta / torch.norm(theta, dim=1, keepdim=True)  

            # Project both sample sets: result shape [num_functions, L]
            nnet_proj = torch.matmul(nnet_samples_slice, theta.T)  # [num_functions, L]
            gp_proj = torch.matmul(gp_samples_slice, theta.T)

            # Sort projections along batch dimension (which is rows now)
            nnet_proj_sorted, _ = torch.sort(nnet_proj, dim=0)
            gp_proj_sorted, _ = torch.sort(gp_proj, dim=0)

            # Compute Wasserstein distances per projection: shape [L]
            wasserstein_distance = torch.abs(nnet_proj_sorted - gp_proj_sorted)
            wasserstein_distance = torch.sum(wasserstein_distance ** p, dim=0) 

            # Softmax over L projections
            weights = torch.softmax(wasserstein_distance, dim=0)  # [L]

            # Weighted sum over L projections
            sw = torch.sum(weights * wasserstein_distance)  # scalar

            ise_bsw_value = sw ** (1. / p)

            dist += ise_bsw_value

        return dist

    def optimize(self, net, data_generator, n_data, num_iters, output_dim, lambd,
                n_samples=128, lr=1e-1, print_every=100, save_ckpt_every = 500):
        if torch.cuda.is_available():
            net = torch.compile(net)
        sdist_hist = np.array([])
        pruned_hist = np.array([])
        prior_optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
        # Cosine Annealing scheudle for the optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(prior_optimizer, T_max=num_iters)
        omega_params = []
        # We want to impose different penalization scalars on different layers
        for layer in net.layers + [net.output_layer]:
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
            # Draw GP samples
            gp_samples = self.gp.sample_functions(
                X.double(), n_samples).detach().float().to(self.device)
            if output_dim > 1:
                gp_samples = gp_samples.squeeze()
            # Draw functions from BNN
            nnet_samples = net.sample_functions(
                X, n_samples).float().to(self.device)
            if output_dim > 1:
                nnet_samples = nnet_samples.squeeze()
            prior_optimizer.zero_grad()
            # Add regularization
            if omega_params:
                reg_loss_vector = torch.tensor([torch.sum(omega_params_per_layer ** 2) \
                                    for omega_params_per_layer in omega_params]).to(self.device)
                regularization = torch.matmul(lambd, reg_loss_vector)
            else:
                regularization = 0
            sdist = self.ebswd(nnet_samples, gp_samples, output_dim = output_dim, device = self.device)
            loss = sdist + regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(omega_params, 100.)
            prior_optimizer.step()
            scheduler.step()

            # Check the number of pruned parameters
            num_pruned = 0
            if omega_params:
                for layer in net.layers:
                    if isinstance(layer, nn.Sequential):
                        for l in layer:
                            if not isinstance(l, nn.BatchNorm2d):
                                num_pruned += l.get_num_pruned_W_std()
                                num_pruned += l.get_num_pruned_b_std()
                    else:
                        num_pruned += (layer.get_W_std() == 0).sum().item()
                        num_pruned += (layer.get_b_std() == 0).sum().item()
                if not self.out_det:
                    num_pruned += (net.output_layer.get_W_std()==0).sum().item()
                    num_pruned += (net.output_layer.get_b_std()==0).sum().item()
            pruned_hist = np.append(pruned_hist, num_pruned)

            sdist_hist = np.append(sdist_hist, float(sdist))
            if (it % print_every == 0) or it == 1:
                print(">>> Iteration # {:3d}: "
                          "Energy-Based Sliced Wasserstein Distance {:.4f}".format(
                            it, float(sdist)), f"Number of pruned stochastic weights: {num_pruned}")
            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                if omega_params:
                    checkpoint = {}
                    for i, layer in enumerate(net.layers):
                        if isinstance(layer, nn.Sequential):
                            W_std = []
                            b_std = []
                            for l in layer:
                                if not isinstance(l, nn.BatchNorm2d):
                                    W_std_result = l.get_W_std()
                                    if isinstance(W_std_result, list):
                                        W_std.extend(W_std_result)
                                    else:
                                        W_std.append(W_std_result)

                                    b_std_result = l.get_b_std()
                                    if isinstance(b_std_result, list):
                                        b_std.extend(b_std_result)
                                    else:
                                        b_std.append(b_std_result)
                        else:
                            W_std = layer.get_W_std()
                            b_std = layer.get_b_std()
                        checkpoint[f"layer_{i}_W_std"] = W_std
                        checkpoint[f"layer_{i}_b_std"] = b_std
                    if self.out_det:
                        checkpoint["out_b_standdev"] = torch.std(net.output_layer.b, device = self.device).item()
                        checkpoint["out_W_standdev"] = torch.std(net.output_layer.W, device = self.device).item()
                    else:
                        checkpoint["output_layer_W_std"] = net.output_layer.get_W_std()
                        checkpoint["output_layer_b_std"] = net.output_layer.get_b_std()
                    torch.save(checkpoint, path)   
                else:
                    torch.save(net.state_dict(), path)
            torch.cuda.empty_cache()
        return pruned_hist, sdist_hist