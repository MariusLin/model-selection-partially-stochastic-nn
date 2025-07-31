# Imports and preparations
import torch
import pandas as pd
import numpy as np
import math
import fire
import os
import sys
import pickle
import warnings 
warnings.simplefilter("ignore", UserWarning)

os.chdir("..")
print ("Running partially stochastic regression")

from Prior_optimization.gpr import GPR
from Prior_optimization import kernels, mean_functions
from Partial_stochasticity.Networks.factorized_gaussian_reparam_mlp import FactorizedGaussianMLPReparameterization
from Samplers.likelihoods import LikGaussian
from Prior_optimization.priors import OptimGaussianPrior
from Utilities.rand_generators import MeasureSetGenerator
from Utilities.normalization import normalize_data
from Utilities.exp_utils import get_input_range
from Metrics.sampling import compute_rhat_regression
from Metrics import uncertainty as uncertainty_metrics
from Partial_stochasticity.Networks.mlp_masked import MLPMasked
from Partial_stochasticity.Networks.regression_net_masked import RegressionNetMasked
from Prior_optimization.optimisation_mapper import PriorOptimisationMapper
from Utilities import util
from Utilities.priors import LogNormal

SEED = 123
util.set_seed(SEED)


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

data_dir = "./data/uci/regression"
noise_var = 0.1
n_hidden = 1
activation_fn = "tanh"
# setup depending on dataset
def setup(dataset):
    if dataset in ["song", "daybikesharing", "onlinenews"]:
        n_units = 200
    else:
        n_units = 100
    n_splits = 10
    out_dir = f"./exp/uci/{dataset}/partially_stochastic"
    util.ensure_dir(out_dir)
    return n_splits, n_units, out_dir

# Configurations for the prior optimization
D = 3                                                            # The factorization depth
mapper_batch_size = 128   
num_iters = 3500                                     
prior_opt_configurations = {
    "n_data": mapper_batch_size,                                # The batch size 
    "num_iters": num_iters,                                     # The number of iterations of the prior optimization
    "lambd": (torch.tensor([1.5, 1.2])/D).to(device),           # The regularization parameters for the layers
    "n_samples": 100,                                           # The number of function samples
    "lr": 3e-2,                                                 # The learning rate for the optimizer
    "print_every": 100,                                         # After how many epochs a evaluation should be printed
    "save_ckpt_every": 500                                      # After how many epochs a checkpoint should be saved
}

def prior_optimization(dataset):
    n_splits, n_units, out_dir = setup(dataset)
    masks_list = []
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id+1, dataset))
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
        # Try the Exponential now
        kernel = kernels.RBF(input_dim=input_dim,
                            lengthscales=torch.tensor([lengthscale], dtype=torch.double),
                            variance=torch.tensor([variance], dtype=torch.double), ARD=True)
        # Place hyper-priors on lengthscales and variances
        kernel.lengthscales.prior = LogNormal(
                torch.ones([input_dim]) * math.log(lengthscale),
                torch.ones([input_dim]) * 1.)
        kernel.variance.prior = LogNormal(
                torch.ones([1]) * 0.1,
                torch.ones([1]) * 1.)
        kernel = kernel.to(device)
        # Initialize the GP model
        gp = GPR(X=torch.from_numpy(X_train_), Y=torch.from_numpy(y_train_).reshape([-1, 1]),
                kern=kernel, mean_function=mean)
        gp.likelihood.variance.set(noise_var)
        # Initialize tunable MLP prior
        hidden_dims = [n_units] * n_hidden
        mlp_reparam = FactorizedGaussianMLPReparameterization(input_dim, output_dim,
            hidden_dims, D = D, activation_fn=activation_fn, scaled_variance=True, device=device)
        mlp_reparam = mlp_reparam.to(device)
        # Perform optimization
        mapper = PriorOptimisationMapper(out_dir=saved_dir, device=device, gp = gp, out_det=False).to(device)
        p_hist, loss_hist = mapper.optimize(mlp_reparam, rand_generator, output_dim=output_dim, **prior_opt_configurations)
        path = os.path.join(saved_dir, "loss_values.log")
        if not os.path.isfile(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)
        np.savetxt(path, loss_hist, fmt='%.6e')
        path = os.path.join(saved_dir, "pruned_values.log")
        if not os.path.isfile(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)
        np.savetxt(path, p_hist, fmt='%.6e')
        print("----" * 20)
        masks_list.append(mlp_reparam.get_det_masks())
    # Save the masks
    with open(os.path.join(out_dir, "masks_list.pkl"), "wb") as f:
        pickle.dump(masks_list, f)


# SGHMC Hyper-parameters
# sampling_configs_ps = {
#     "batch_size": 32,            # Mini-batch size
#     "num_samples": 40,           # Total number of samples for each chain
#     "n_discarded": 10,           # Number of the first samples to be discared for each chain
#     "num_burn_in_steps": 1500,   # Number of burn-in steps, 2000 yields better MSE
#     "keep_every": 1500,          # Thinning interval
#     "lr": 0.03,                  # Step size
#     "num_chains": 5,             # Number of chains
#     "mdecay": 0.01,               # Momentum coefficient
#     "print_every_n_samples": 5,  # After how many iterations an evaluation should be printed
#     "lambd": 1e-8,               # The lambda for encouraging sparsity in the deterministic weights
#     "train_det_before": None      # How many iterations the deterministic weights should be trained prior to sampling from the posterior
# }
sampling_configs_ps = {
    "batch_size": 32,
    "num_samples": 40,
    "n_discarded": 10,
    "num_burn_in_steps": 2000,
    "keep_every": 2000,
    "lr": 0.01,
    "num_chains": 4,
    "mdecay": 0.01,
    "print_every_n_samples": 20
}

def train(dataset, train_det, lam, train_steps):
    n_splits, n_units, out_dir = setup(dataset)
    # Load the masks
    with open(os.path.join(out_dir, "masks_list.pkl"), "rb") as f:
        masks_list = pickle.load(f)

    results = {"rmse": [], "nll": []}

    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id+1, dataset))
        saved_dir = os.path.join(out_dir, str(split_id))
        
        # Load the dataset
        X_train, y_train, X_test, y_test = util.load_uci_data(
                data_dir, split_id, dataset)
        X_train_, y_train_, X_test_, y_test_, y_mean, y_std = normalize_data(
                X_train, y_train, X_test, y_test)
        input_dim, output_dim = int(X_train.shape[-1]), 1
        # Initialize the neural network and likelihood modules
        # Load the optimized prior for the MLP
        ckpt_path = os.path.join(out_dir, str(split_id), "ckpts", "it-{}.ckpt".format(num_iters))
        checkpoint = torch.load(ckpt_path)
        W_std_list = []
        b_std_list = []
        W_std_out = 1
        b_std_out = 1
        for key, value in checkpoint.items():
            if key == "out_b_standdev":
                b_std_out = value
            elif key == "out_W_standdev":
                W_std_out = value
            elif "W_std" in key:
                W_std_list.append(value)
            elif "b_std" in key:
                b_std_list.append(value)
        weight_masks, bias_masks = masks_list[split_id]
        net = MLPMasked(input_dim, output_dim, [n_units] * n_hidden, activation_fn, weight_masks, bias_masks, D = D, 
                        prior_W_std_list = W_std_list, prior_b_std_list = b_std_list, 
                        W_std_out = W_std_out, b_std_out = b_std_out, device = device)
        net = net.to(device)
        likelihood = LikGaussian(noise_var)
        
        # Load the optimized prior
        prior = OptimGaussianPrior(ckpt_path)
        
        # Initialize bayesian neural network with SGHMC sampler
        saved_dir = os.path.join(out_dir, str(split_id))
        bayes_net = RegressionNetMasked(net = net, likelihood=likelihood, prior=prior, ckpt_dir= saved_dir, 
                                        n_gpu=n_gpu)
        #Add additional information to the dictinoary
        sampling_configs_ps["lambd"]= lam
        sampling_configs_ps["train_det"] = train_det
        sampling_configs_ps["det_train_steps"] = train_steps
  
        # Start sampling
        bayes_net.sample_multi_chains(X_train_, y_train_, **sampling_configs_ps)
        pred_mean, pred_var, preds, raw_preds = bayes_net.predict(X_test_, True, True)
        r_hat = compute_rhat_regression(raw_preds, sampling_configs_ps["num_chains"])
        print("R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))

        rmse = uncertainty_metrics.rmse(pred_mean, y_test_)
        nll = uncertainty_metrics.gaussian_nll(y_test_, pred_mean, pred_var)
        print("> RMSE = {:.4f} | NLL = {:.4f}".format(rmse, nll))
        results['rmse'].append(rmse)
        results['nll'].append(nll)
        result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(out_dir, f"optim_results_{train_det}_{lam}_{train_steps}.csv"), sep="\t", 
                     index=False)
    print("Final results")
    print("> RMSE: mean {:.4e} $\pm$ {:.4e} | NLL: mean {:.4e} $\pm$ {:.4e}".format(
            float(result_df['rmse'].mean()), float(result_df['rmse'].std()),
            float(result_df['nll'].mean()), float(result_df['nll'].std())))

def run(dataset, train_det, lam, train_steps):
    prior_optimization(dataset)
    train(dataset, train_det, lam, train_steps)


def run_training_only(dataset, train_det, lam, train_steps):
    train(dataset, train_det, lam, train_steps)

def help(dataset):
    lambd = 5e-10
    det_train_steps = 0
    det_train_time = "during" #"during", "after", "before"
    print (f"Dataset is: {dataset}")
    print (f"Training of deterministic weights is {det_train_time} inference")
    print (f"Training of deterministic weights for {det_train_steps} steps")
    print (f"The Lambda for encouraging sparsity is {lambd}")
    run_training_only(dataset, "during",lambd, det_train_steps)
    #run(dataset, "during",lambd, det_train_steps)

def main():
    fire.Fire(help)
main()