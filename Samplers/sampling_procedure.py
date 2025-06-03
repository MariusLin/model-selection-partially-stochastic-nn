import torch
import torch.nn as nn
from Utilities import util
import copy
from itertools import islice
import numpy as np

from Samplers.sghmc import SGHMC
from Samplers.adaptive_sghmc import AdaptiveSGHMC
from Metrics.uncertainty import gaussian_nll, rmse
from Layers.partially_stochastic_linear_not_pruned import PartiallyStochasticLinearNP



class Mapper_Sampler:
    def __init__(self, likelihood, adapted):
          self.step = 0
          self.lik_module = likelihood
          self.num_samples = 0
          self.sampled_weights = []
          self.adapted = adapted
    

    def sample_multi_chains(self, net, data_loader, num_datapoints, X= None, y = None, num_chains=1, keep_every=200,
                            n_discarded=10, num_burn_in_steps=2000, num_samples = 30,
                            print_every_n_samples=5):
        """
        Use multiple chains of sampling.

        Args:
            data_loader: instance of DataLoader, the dataloader for training
                data. Notice that we have to choose either numpy arrays or
                dataloader for the input data.
            num_samples: int, number of set of parameters per chain
                we want to sample.
            num_chains: int, number of chains.
            keep_every: number of sampling steps (after burn-in) to perform
                before keeping a sample.
            n_discarded: int, the number of first samples will
                be discarded.
            num_burn_in_steps: int, number of burn-in steps to perform.
                This value is passed to the given `sampler` if it supports
                special burn-in specific behavior like that of Adaptive SGHMC.
            lr: float, learning rate.
            batch_size: int, batch size.
            epsilon: float, epsilon for numerical stability.
            mdecay: float, momemtum decay.
            print_every_n_samples: int, defines after how many samples we want
                to print out the statistics of the sampling process.
            continue_training: bool, defines whether we want to continue
                from the last training run.
            resample_prior_every: int, num ber of sampling steps to perform
                before resampling prior.
        """
        for chain in range(num_chains):
            print("Chain: {}".format(chain+1))
            net.reset_parameters()
            self.sample_one_chain(net, data_loader, num_datapoints, X = X, y = y, keep_every = keep_every, num_burn_in_steps = num_burn_in_steps, n_discarded = n_discarded, num_samples = num_samples,
                       print_every_n_samples = print_every_n_samples)

    def sample_one_chain(self, net, dataloader, num_datapoints, X = None, y = None,
                         keep_every = 200, num_burn_in_steps = 8000, n_discarded = 10, num_samples = 30,
                         print_every_n_samples = 5):
        #Initialize a data loader for training data.
        train_loader = util.inf_loop(dataloader)
        # Estimate the number of update steps
        num_steps = 0 if num_samples is None else (num_samples+1) * keep_every

        num_steps += num_burn_in_steps
        # Initialize the batch generator
        batch_generator = islice(enumerate(train_loader), num_steps)

        params = []
        for layer in net.layers:
            for name, param in layer.named_parameters():
                if name.endswith("std") or name.endswith("mu"):
                    params.append(param)
    
        for param in params:
            param.requires_grad_(True)
        if self.adapted:
            sampler = AdaptiveSGHMC(params)
        else:
            sampler = SGHMC(params)
        # Start sampling
        net.train()
        n_samples = 0 # used to discard first samples
        for step, (x_batch, y_batch) in batch_generator:
            x_batch = x_batch.view(y_batch.shape[0], -1)
            y_batch = y_batch.view(-1, 1)
                
            fx_batch = net(x_batch).view(-1, 1)

            sampler.zero_grad()

            # Calculate the negative log joint density
            loss = self._neg_log_joint(net, fx_batch, y_batch, num_datapoints).float()
            print (loss.device)
            for name, param in net.named_parameters():
                print (f"{name}:{param.device}")
            # Estimate the gradients
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 100.)

            # Update parameters
            sampler.step()
            self.step += 1
            # Save the sampled weight
            if (step > num_burn_in_steps) and \
                        ((step - num_burn_in_steps) % keep_every == 0):
                n_samples += 1
                if n_samples > n_discarded:
                    for layer in net.layers:
                        if isinstance(layer, PartiallyStochasticLinearNP):   
                            self.sampled_weights.append(layer.stochastic_mu + torch.rand_like(layer.stochastic_mu)\
                                                        *layer.stochastic_std)
                    self.num_samples += 1
                    # Print evaluation on training data
                    if self.num_samples % print_every_n_samples == 0:
                        net.eval()
                        if (X is not None) and (y is not None):
                            self._print_evaluations(net, X.cpu(), y.cpu(), self.num_samples, self.sampled_weights, True)
                        else:
                            self._print_evaluations(net, x_batch.cpu(), y_batch.cpu(), self.num_samples, self.sampled_weights, True)
                        net.train()



    def _print_evaluations(self, net, x, y, num_samples, sampled_weights, train=True):
        """Evaluate the sampled weights on training/validation data and
            during the training log the results.

        Args:
            x: numpy array, shape [batch_size, num_features], the input data.
            y: numpy array, shape [batch_size, 1], the corresponding targets.
            train: bool, indicate whether we're evaluating on the training data.
        """
        net.eval()
        pred_mean, pred_var = net.predict(x, sampled_weights)
        total_nll = gaussian_nll(y, pred_mean, pred_var)
        y = np.asarray(y)
        total_rmse = rmse(pred_mean, y)

        if train:
            print("Samples # {:5d} : NLL = {:11.4e} "
                    "RMSE = {:.4e} ".format(num_samples, total_nll,
                                            total_rmse))
        else:
            print("Validation: NLL = {:11.4e} RMSE = {:.4e}".format(
                    total_nll, total_rmse))

        net.train()    

    def _neg_log_joint(self, net, fx_batch, y_batch, num_datapoints):
        """Calculate model's negative log joint density.

            Note that the gradient is computed by: g_prior + N/n sum_i grad_theta_xi.
            Because of that we divide here by N=num of datapoints
            since in the sample we will rescale the gradient by N again.

        Args:
            fx_batch: torch tensor, the predictions.
            y_batch: torch tensor, the corresponding targets.
            num_datapoints: int, the number of data points in the entire
                training set.

        Return:
            The negative log joint density.
        """
        log_p_prior = 0
        for layer in net.layers:
            if isinstance(layer, PartiallyStochasticLinearNP):             
                param = layer.stochastic_mu
                mu = layer.initial_mu
                var = layer.stochastic_std **2
                log_p_prior -= torch.sum(((param - mu) ** 2) / (2 * var)) 
        return (self.lik_module(fx_batch, y_batch)) / y_batch.shape[0] + log_p_prior / num_datapoints
    
    def reset_sampled_weights(self):
        self.sampled_weights = []
        self.num_samples = 0