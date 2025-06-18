# Imports and preparations
import torch
import pandas as pd
import numpy as np
import math
import fire
import os

os.chdir("..")

from Prior_optimization.gpr import GPR
from Prior_optimization import kernels, mean_functions, priors
from Full_stochasticity.Networks.gaussian_reparam_mlp import GaussianMLPReparameterization
from Full_stochasticity.Networks.mlp import MLP
from Samplers.likelihoods import LikGaussian
from Prior_optimization.priors import OptimGaussianPrior
from Prior_optimization.wasserstein_mapper import MapperWasserstein
from Utilities.rand_generators import MeasureSetGenerator
from Utilities.normalization import normalize_data
from Utilities.exp_utils import get_input_range
from Metrics.sampling import compute_rhat_regression
from Metrics import uncertainty as uncertainty_metrics
from Full_stochasticity.Networks.regression_net import RegressionNet
from Utilities import util

SEED = 123
util.set_seed(SEED)

import warnings 
warnings.simplefilter("ignore", UserWarning)


# setting device on GPU if available, else CPU
n_gpu = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device} \n')

#Additional Info when using cuda
if device.type == 'cuda':
    n_gpu += torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('Number of available GPUs:', str(n_gpu))

# Setups are all the same
data_dir = "./data/uci/regression"
noise_var = 0.1
n_hidden = 1
activation_fn = "tanh"
num_iters = 200  # Number of iteterations of Wasserstein optimization
lr = 0.05        # The learning rate
n_samples = 128  # The mini-batch size
# setup depending on dataset
def setup(dataset):
    if dataset == "protein":
        n_splits = 5
        n_units = 200
    else:
        n_splits = 10
        n_units = 100
    out_dir = f"./exp/uci/{dataset}/fully_stochastic"
    util.ensure_dir(out_dir)
    return n_splits, n_units, out_dir

# Here prior optimization
def prior_optimization(dataset):
    n_splits, n_units, out_dir = setup(dataset)
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id, dataset))
        # Load the dataset
        saved_dir = os.path.join(out_dir, str(split_id))
        X_train, y_train, X_test, y_test = util.load_uci_data(
                data_dir, split_id, dataset)
        X_train_, y_train_, X_test_, y_test_, y_mean, y_std = normalize_data(
                X_train, y_train, X_test, y_test)
        x_min, x_max = get_input_range(X_train_, X_test_)
        input_dim, output_dim = int(X_train.shape[-1]), 1
        
        # Initialize the measurement set generator
        rand_generator = MeasureSetGenerator(X_train_, x_min, x_max, 0.7)
        
        # Initialize the mean and covariance function of the target hierarchical GP prior
        mean = mean_functions.Zero()
        
        lengthscale = math.sqrt(2. * input_dim)
        variance = 1.
        kernel = kernels.RBF(input_dim=input_dim,
                            lengthscales=torch.tensor([lengthscale], dtype=torch.double),
                            variance=torch.tensor([variance], dtype=torch.double), ARD=True)

        # Place hyper-priors on lengthscales and variances
        kernel.lengthscales.prior = priors.LogNormal(
                torch.ones([input_dim]) * math.log(lengthscale),
                torch.ones([input_dim]) * 1.)
        kernel.variance.prior = priors.LogNormal(
                torch.ones([1]) * 0.1,
                torch.ones([1]) * 1.)
            
        # Initialize the GP model
        gp = GPR(X=torch.from_numpy(X_train_), Y=torch.from_numpy(y_train_).reshape([-1, 1]),
                kern=kernel, mean_function=mean)
        gp.likelihood.variance.set(noise_var)
        
        # Initialize tunable MLP prior
        hidden_dims = [n_units] * n_hidden
        mlp_reparam = GaussianMLPReparameterization(input_dim, output_dim,
            hidden_dims, activation_fn, scaled_variance=True)
        
        mapper = MapperWasserstein(gp, mlp_reparam, rand_generator, out_dir=saved_dir,
                                output_dim=output_dim, n_data=100,
                                wasserstein_steps=(0, 200),
                                wasserstein_lr=0.02,
                                logger=None, wasserstein_thres=0.1,
                                n_gpu=0, gpu_gp=False)
        
        w_hist = mapper.optimize(num_iters=num_iters, n_samples=n_samples,
                                lr=lr, print_every=10, save_ckpt_every=10, debug=True)
        path = os.path.join(saved_dir, "wsr_values.log")
        np.savetxt(path, w_hist, fmt='%.6e')
        print("----" * 20)

# Here training / evaluation

# Configure the SGHMC sampler
sampling_configs = {
    "batch_size": 32,
    "num_samples": 40,
    "n_discarded": 10,
    "num_burn_in_steps": 2000,
    "keep_every": 2000,
    "lr": 1e-2,
    "num_chains": 4,
    "mdecay": 1e-2,
    "print_every_n_samples": 5
}
def train(dataset):
    n_splits, n_units, out_dir = setup(dataset)
    results = {"rmse": [], "nll": []}
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id, dataset))
        saved_dir = os.path.join(out_dir, str(split_id))
        
        # Load the dataset
        X_train, y_train, X_test, y_test = util.load_uci_data(
                data_dir, split_id, dataset)
        input_dim, output_dim = int(X_train.shape[-1]), 1
        
        # Initialize the neural network and likelihood modules
        net = MLP(input_dim, output_dim, [n_units] * n_hidden, activation_fn)
        likelihood = LikGaussian(noise_var)
        
        # Load the optimized prior
        ckpt_path = os.path.join(out_dir, str(split_id), "ckpts", "it-{}.ckpt".format(num_iters))
        prior = OptimGaussianPrior(ckpt_path)
        
        # Initialize bayesian neural network with SGHMC sampler
        saved_dir = os.path.join(out_dir, str(split_id))
        bayes_net = RegressionNet(net, likelihood, prior, saved_dir, n_gpu=0)
        
        # Start sampling
        bayes_net.sample_multi_chains(X_train, y_train, **sampling_configs)
        pred_mean, pred_var, preds, raw_preds = bayes_net.predict(X_test, True, True)
        r_hat = compute_rhat_regression(raw_preds, sampling_configs["num_chains"])
        print("R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))

        rmse = uncertainty_metrics.rmse(pred_mean, y_test)
        nll = uncertainty_metrics.gaussian_nll(y_test, pred_mean, pred_var)
        print("> RMSE = {:.4f} | NLL = {:.4f}".format(rmse, nll))
        results['rmse'].append(rmse)
        results['nll'].append(nll)


        result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(out_dir, "optim_results.csv"), sep="\t", index=False)

    print("Final results")
    print("> RMSE: mean {:.4e}; std {:.4e} | NLL: mean {:.4e} std {:.4e}".format(
            float(result_df['rmse'].mean()), float(result_df['rmse'].std()),
            float(result_df['nll'].mean()), float(result_df['nll'].std())))

def run(dataset):
    prior_optimization(dataset)
    train(dataset)
# everything in one function where only calling with different arguments

def run_training_only(dataset):
    train(dataset)

def main():
    fire.Fire(run)

main()