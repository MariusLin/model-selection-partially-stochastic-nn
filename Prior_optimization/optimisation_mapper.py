import numpy as np
import torch
from Utilities.util import ensure_dir
import os


class PriorOptimisationMapper():
    def __init__(self, out_dir, gp, device = "cpu"):
        self.out_dir = out_dir
        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)
        self.device = device
        self.gp = gp

    def to(self, device):
        self.device = torch.device(device)
        return self
    # ISEBSW from Nguyen, Ho 2023
    # def ISEBSW(self, X, Y, L=10, p=2, device="cpu"):
    #     def rand_projections(dim, num_projections=1000,device='cpu'):
    #         projections = torch.randn((num_projections, dim),device=device)
    #         projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    #         return projections

    #     def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    #         X_prod = torch.matmul(X, theta.transpose(0, 1))
    #         Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    #         X_prod = X_prod.view(X_prod.shape[0], -1)
    #         Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    #         wasserstein_distance = torch.abs(
    #             (
    #                     torch.sort(X_prod, dim=0)[0]
    #                     - torch.sort(Y_prod, dim=0)[0]
    #             )
    #         )
    #         wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    #         return wasserstein_distance
    #     dim = X.size(1)
    #     theta = rand_projections(dim, L,device)
    #     wasserstein_distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    #     wasserstein_distances =  wasserstein_distances.view(1,L)
    #     weights = torch.softmax(wasserstein_distances,dim=1)
    #     sw = torch.sum(weights*wasserstein_distances,dim=1).mean()
    #     return  torch.pow(sw,1./p)
    
    def expectation_ISEBSW(self, nnet_samples, gp_samples, output_dim,  L=10, p=2, device="cpu"):
        """
        Vectorized computation of expectation ISEBSW over the functions.
        
        nnet_samples: tensor of shape [batch_size, num_functions, output_dim]
        gp_samples: tensor of shape [batch_size, num_functions, output_dim]
        """
        # Move num_functions to batch dimension: [num_functions, batch_size, output_dim]
        nnet_samples = nnet_samples.permute(1, 0, 2).contiguous()
        gp_samples = gp_samples.permute(1, 0, 2).contiguous()

        # Generate random projections: shared across all functions
        theta = torch.randn((L, output_dim), device=device)
        theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))  # normalize
        # Project nnet and gp samples: [num_functions, batch_size, L]
        nnet_proj = torch.matmul(nnet_samples, theta.T) 
        gp_proj = torch.matmul(gp_samples, theta.T)

        # Sort projections along batch dimension
        nnet_proj_sorted, _ = torch.sort(nnet_proj, dim=1)
        gp_proj_sorted, _ = torch.sort(gp_proj, dim=1)

        # Compute Wasserstein distances: [num_functions, L]
        wasserstein_distance = torch.abs(nnet_proj_sorted - gp_proj_sorted)
        wasserstein_distance = torch.sum(wasserstein_distance ** p, dim=1)

        # Softmax weighting across L projections: [num_functions, L]
        weights = torch.softmax(wasserstein_distance, dim=1)

        # Weighted sum for each function: [num_functions]
        sw = torch.sum(weights * wasserstein_distance, dim=1)

        # Compute SW^1/p
        ise_bsw_values = sw ** (1. / p)

        # Return mean across all functions
        return ise_bsw_values.mean()
    

    # def energy_distance(self, bnn_samples, gp_samples, num_pairs=64):
    #     """
    #     Approximate Energy Distance between BNN and GP samples using subsampling.
        
    #     bnn_samples, gp_samples: shape [batch_size, num_functions, output_dim]
    #     num_pairs: number of random pairs to sample for each term
    #     """
    #     batch_size, num_functions, output_dim = bnn_samples.shape

    #     # Flatten batch and num_functions for easy indexing
    #     indices = torch.randint(0, num_functions, (batch_size, num_pairs), device=bnn_samples.device)

    #     # Sample random pairs for cross term
    #     idx1 = torch.randint(0, num_functions, (batch_size, num_pairs), device=bnn_samples.device)
    #     idx2 = torch.randint(0, num_functions, (batch_size, num_pairs), device=bnn_samples.device)

    #     # Gather samples
    #     X1 = torch.gather(bnn_samples, 1, idx1.unsqueeze(-1).expand(-1, -1, output_dim))
    #     X2 = torch.gather(bnn_samples, 1, idx2.unsqueeze(-1).expand(-1, -1, output_dim))

    #     Y1 = torch.gather(gp_samples, 1, idx1.unsqueeze(-1).expand(-1, -1, output_dim))
    #     Y2 = torch.gather(gp_samples, 1, idx2.unsqueeze(-1).expand(-1, -1, output_dim))

    #     # Compute pairwise distances for each term
    #     d_xy = torch.norm(X1 - Y1, dim=-1).mean(dim=1)  # cross term
    #     d_xx = torch.norm(X1 - X2, dim=-1).mean(dim=1)  # BNN self-term
    #     d_yy = torch.norm(Y1 - Y2, dim=-1).mean(dim=1)  # GP self-term

    #     energy = 2.0 * d_xy - d_xx - d_yy
    #     return energy.mean()


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
            # Draw GP samples
            gp_samples = self.gp.sample_functions(
                X.double(), n_samples).detach().float().to(self.device)
            if output_dim > 1:
                gp_samples = gp_samples.squeeze()
            # Draw functions from Neural Network
            nnet_samples = net.sample_functions(
                X, n_samples).float().to(self.device)
            if output_dim > 1:
                nnet_samples = nnet_samples.squeeze()
            prior_optimizer.zero_grad()
            reg_loss_vector = torch.tensor([torch.sum(omega_params_per_layer ** 2) \
                               for omega_params_per_layer in omega_params]).to(self.device)
            regularization = torch.matmul(lambd, reg_loss_vector)
            sdist = self.expectation_ISEBSW(nnet_samples, gp_samples, output_dim = output_dim, device = self.device)
            loss = sdist + regularization
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
                          "Energy-Based Sliced Wasserstein Distance {:.4f}".format(
                            it, float(sdist)), f"Number of pruned stochastic weights: {num_pruned}")
            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "ps-it-{}.ckpt".format(it))
                checkpoint = {}
                for i, layer in enumerate(net.layers):
                    W_std = layer.get_W_std()
                    b_std = layer.get_b_std()
                    checkpoint[f"layer_{i}_W_std"] = W_std
                    checkpoint[f"layer_{i}_b_std"] = b_std
                torch.save(checkpoint, path)
        return pruned_hist, sdist_hist