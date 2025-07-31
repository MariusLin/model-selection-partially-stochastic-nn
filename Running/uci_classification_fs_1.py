# Imports and preparations
import torch
import pandas as pd
import numpy as np
import math
import fire
import pickle
import os
import torch.utils.data as data_utils

os.chdir("..")
print ("Running fully stochastic classification with sparse BNN")

from Prior_optimization.gpr import GPR
from Prior_optimization import kernels, mean_functions
from Full_stochasticity.Networks.factorized_gaussian_reparam_mlp import FactorizedGaussianMLPReparameterization
from Full_stochasticity.Networks.masked_mlp import MLPMasked
from Samplers.likelihoods import LikCategorical
from Prior_optimization.priors import OptimGaussianPrior
from Prior_optimization.optimisation_mapper import PriorOptimisationMapper
from Utilities.rand_generators import MeasureSetGenerator
from Utilities.normalization import normalize_data
from Utilities.exp_utils import get_input_range
from Metrics.sampling import compute_rhat_classification
from Full_stochasticity.Networks.classification_net import ClassificationNet
from Utilities import util
from Utilities.priors import LogNormal
import Metrics.metrics_tensor as metrics_classification
import Metrics.uncertainty as uncertainty_metrics

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
data_dir = "./data/uci/classification"
noise_var = 0.1
n_hidden = 1
activation_fn = "tanh"
num_iters = 3500
# setup depending on dataset
def setup(dataset):
    print (dataset)
    n_splits = 10
    n_units = 100
    out_dir = f"./exp/uci/{dataset}/fully_stochastic_sparse_level3"
    util.ensure_dir(out_dir)
    return n_splits, n_units, out_dir

def dataset_settings(dataset):
    if dataset == "banknote":
        num_classes = 2
    elif dataset == "credit":
        num_classes = 2
    elif dataset == "drybean":
        num_classes = 7
    elif dataset == "maternalhealth":
        num_classes = 3
    elif dataset == "obesity":
        num_classes = 7
    elif dataset == "onlineshoppers":
        num_classes = 2
    elif dataset == "steel":
        num_classes = 3
    elif dataset == "htru2":
        num_classes = 2
    elif dataset == "sepsis":
        num_classes = 2 
    return num_classes 

# Configurations for the prior optimization
D = 3
mapper_batch_size = 256                                         # The factorization depth
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
    num_classes = dataset_settings(dataset)
    masks_list = []
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id+1, dataset))
        # Load the dataset
        saved_dir = os.path.join(out_dir, str(split_id))
        X_train, y_train, X_test, y_test = util.load_uci_data(
                data_dir, split_id, dataset)
        X_train_, y_train_, X_test_, y_test_, y_mean, y_std = normalize_data(
                X_train, y_train, X_test, y_test)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        x_min, x_max = get_input_range(X_train_, X_test_)
        input_dim = int(X_train.shape[-1])
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
        kernel.lengthscales.prior = LogNormal(
                torch.ones([input_dim]) * math.log(lengthscale),
                torch.ones([input_dim]) * 1.)
        kernel.variance.prior = LogNormal(
                torch.ones([1]) * 0.1,
                torch.ones([1]) * 1.)
        kernel = kernel.to(device)
        # Initialize the GP model
        gp = GPR(X=torch.from_numpy(X_train_), Y=util.to_one_hot(torch.from_numpy(y_train), num_classes),
                kern=kernel, mean_function=mean)
        gp.likelihood.variance.set(noise_var)
        # Initialize tunable MLP prior
        hidden_dims = [n_units] * n_hidden
        mlp_reparam = FactorizedGaussianMLPReparameterization(input_dim, num_classes,
            hidden_dims, D = D, activation_fn=activation_fn, scaled_variance=True, device=device)
        mlp_reparam = mlp_reparam.to(device)
        # Perform optimization
        mapper = PriorOptimisationMapper(out_dir=saved_dir, device=device, gp = gp, out_det = False).to(device)
        p_hist, loss_hist = mapper.optimize(mlp_reparam, rand_generator, output_dim=num_classes, **prior_opt_configurations)
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
    num_classes = dataset_settings(dataset)
    with open(os.path.join(out_dir, "masks_list.pkl"), "rb") as f:
        masks_list = pickle.load(f)
    results = {"acc": [], "nll": []}
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id, dataset))
        saved_dir = os.path.join(out_dir, str(split_id))
        
        # Load the dataset
        X_train, y_train, X_test, y_test = util.load_uci_data(
                data_dir, split_id, dataset)
        input_dim = int(X_train.shape[-1])
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test).long()
        data_loader = data_utils.DataLoader(
            data_utils.TensorDataset(X_train, y_train),
            batch_size=sampling_configs["batch_size"], shuffle=True)
        
        # Initialize the neural network and likelihood modules
        # Load the optimized prior for the MLP
        ckpt_path = os.path.join(out_dir, str(split_id), "ckpts", "it-{}.ckpt".format(num_iters))
        checkpoint = torch.load(ckpt_path)
        W_std_list = []
        b_std_list = []
        b_std_out = 1
        W_std_out = 1
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
        net = MLPMasked(input_dim, num_classes, [n_units] * n_hidden, activation_fn, weight_masks, bias_masks, 
                        b_std = b_std_out, W_std = W_std_out, task = "classification", device = device)
        net = net.to(device)
        likelihood = LikCategorical()
        # Load the optimized prior for the classification net
        prior = OptimGaussianPrior(ckpt_path)
        # Initialize bayesian neural network with SGHMC sampler
        saved_dir = os.path.join(out_dir, str(split_id))
        bayes_net = ClassificationNet(net, likelihood, prior, saved_dir, n_gpu=n_gpu)
        
        # Start sampling
        bayes_net.sample_multi_chains(data_loader= data_loader, **sampling_configs)
        pred_mean, raw_preds = bayes_net.predict(X_test, True, True)
        p = raw_preds.cpu().numpy()
        p_mean = pred_mean.cpu().numpy()
        r_hat = compute_rhat_classification(p, sampling_configs["num_chains"])
        print("R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))

        acc = uncertainty_metrics.accuracy(p_mean, y_test)
        nll = metrics_classification.nll(pred_mean, y_test)
        print("> Accuracy = {:.4f} | NLL = {:.4f}".format(acc, nll))
        results['acc'].append(acc.cpu().item() if torch.is_tensor(acc) else acc)
        results['nll'].append(nll.cpu().item() if torch.is_tensor(nll) else nll)

        result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(out_dir, "optim_results.csv"), sep="\t", index=False)

    print("Final results")
    print("> Accuracy: mean {:.4e} $\pm$ {:.4e} | NLL: mean {:.4e} $\pm$ {:.4e}".format(
            float(result_df['acc'].mean()), float(result_df['acc'].std()),
            float(result_df['nll'].mean()), float(result_df['nll'].std())))

def run(dataset):
    prior_optimization(dataset)
    train(dataset)

def run_training_only(dataset):
    train(dataset)

def main():
    fire.Fire(run)
    #fire.Fire(run_training_only)

main()