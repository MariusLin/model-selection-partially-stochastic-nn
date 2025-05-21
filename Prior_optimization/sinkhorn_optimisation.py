import numpy as np
import torch
from geomloss import SamplesLoss
from Utilities.util import ensure_dir
import os


class SinkhornMapper():
    def __init__(self, out_dir, device = "cpu"):
        self.out_dir = out_dir
        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)
        self.device = device


    def to(self, device):
        self.device = torch.device(device)
        return self 


    def calculate (self, nnet_samples, gp_samples, blur):
        sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur) 
        if nnet_samples.shape[2] == 1:
            return sinkhorn_loss(nnet_samples.squeeze(), gp_samples.squeeze())
        else:
            return sinkhorn_loss(nnet_samples.contiguous(), gp_samples.contiguous()).mean()


    def optimize_sparse(self, net, gp, data_generator, n_data, num_iters, output_dim, D, lambd, X_train, y_train,
                n_samples=128, lr=1e-1, print_every=10, gpu_gp = True, save_ckpt_every = 50):
        
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
                                                        for omega_params_per_layer in omega_params])
            regularization = torch.matmul(lambd, reg_loss_vector)
            # Compute blur for Sinkhorn Distance
            t = (10-0.001*num_iters)/(1- num_iters)
            m = 0.001-t
            blur = m*it +t
            # Compute the Sinkhorn Distance
            sdist = self.calculate(nnet_samples, gp_samples, blur)
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
                                    "Sinkhorn Dist {:.4f}".format(
                                        it, float(sdist)), f"Number of pruned stochastic weights: {num_pruned}")
            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "sparse-it-{}.ckpt".format(it))
                torch.save(net.state_dict(), path)

        return pruned_hist, sdist_hist
    

    def optimize_not_sparse(self, net, gp, data_generator, n_data, num_iters, output_dim, X_train, y_train, 
                    n_samples=128, lr=1e-1, print_every=10, gpu_gp = True, save_ckpt_every = 50):
        
        sdist_hist = np.array([])
        y_train_shape = y_train.shape
        y_train = y_train.unsqueeze(2).expand(y_train_shape[0], n_samples, y_train_shape[1])
        prior_optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
    
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
            # Compute blur for Sinkhorn Distance
            t = (10-0.001*num_iters)/(1- num_iters)
            m = 0.001-t
            blur = m*it +t
            # Compute the Sinkhorn Distance
            sdist = self.calculate(nnet_samples, gp_samples, blur)
            sdist.backward()
            prior_optimizer.step()

            sdist_hist = np.append(sdist_hist, float(sdist))
            if (it % print_every == 0) or it == 1:
                print(">>> Iteration # {:3d}: "
                                    "Sinkhorn Dist {:.4f}".format(
                                        it, float(sdist)))
            # Save checkpoint
            if ((it) % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "not-sparse-it-{}.ckpt".format(it))
                torch.save(net.state_dict(), path)

        return sdist_hist