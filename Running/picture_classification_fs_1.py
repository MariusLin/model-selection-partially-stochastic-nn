# Imports and preparations
import torch
import pandas as pd
import numpy as np
import math
import fire
import pickle
import os
import torch.utils.data as data_utils
from torch.utils.data import random_split
import torchvision
from torchvision.transforms import transforms
import csv

os.chdir("..")
print ("Running fully stochastic sparse CNN")

from Prior_optimization.gpr import GPR
from Prior_optimization import kernels, mean_functions
from Full_stochasticity.Networks.factorized_gaussian_reparam_preresnet import FactorizedGaussianPreResNetReparameterization
from Full_stochasticity.Networks.masked_preresnet import MaskedPreResNet
from Samplers.likelihoods import LikCategorical
from Prior_optimization.priors import OptimGaussianPrior
from Prior_optimization.optimisation_mapper import PriorOptimisationMapper
from Utilities.rand_generators import ClassificationGenerator
from Metrics.sampling import compute_rhat_classification
from Full_stochasticity.Networks.classification_net import ClassificationNet
from Utilities import util
from Utilities.priors import LogNormal
import Metrics.metrics_tensor as metrics_classification
import Metrics.uncertainty as uncertainty_metrics

SEED = 123
util.set_seed(SEED)
torch.manual_seed(SEED)

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
data_dir = "./data/vision"
num_iters = 3500
# setup depending on dataset
def setup(dataset):
    out_dir = f"./exp/vision/{dataset}/fully_stochastic_sparse"
    util.ensure_dir(out_dir)
    return out_dir

def data_splits(dataset_name):
    data_dir = f"data/vision/{dataset_name}"
    util.ensure_dir(data_dir)
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # Load the CIFAR-10 training dataset
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        # Define the sizes for the training and test sets
        dataset_size = len(dataset)
        split_size = dataset_size // 10  # Size of each split
        # Ensure that the dataset size is divisible by 10
        remainder = dataset_size % 10
        if remainder != 0:
            split_size += 1  # Add the remainder to the last split
        splits = random_split(dataset, [split_size] * 10)

        train_test_splits = []

        for split in splits:
            split_len = len(split)
            train_len = int(split_len * 0.9)
            test_len = split_len - train_len  # Ensure the total length matches
            train_subset, test_subset = random_split(split, [train_len, test_len])
            train_test_splits.append((train_subset, test_subset))

    elif dataset_name == "MNIST":
        # Define a transform to normalize the data
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))])

        # Load the MNIST training dataset
        dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        # Define the sizes for the training and test sets
        dataset_size = len(dataset)
        split_size = dataset_size // 10  # Size of each split
        # Ensure that the dataset size is divisible by 10
        remainder = dataset_size % 10
        if remainder != 0:
            split_size += 1  # Add the remainder to the last split
        splits = random_split(dataset, [split_size] * 10)

        train_test_splits = []

        for split in splits:
            split_len = len(split)
            train_len = int(split_len * 0.9)
            test_len = split_len - train_len  # Ensure the total length matches
            train_subset, test_subset = random_split(split, [train_len, test_len])
            train_test_splits.append((train_subset, test_subset))

    return train_test_splits, dataset

# Extract features and labels from dataset using indices
def extract_features_and_labels(indices, dataset):
    features, labels = zip(*[dataset[i] for i in indices])
    return torch.stack(features), torch.tensor(labels)
# Configurations for the prior optimization
D = 3
mapper_batch_size = 32                                         # The factorization depth
prior_opt_configurations = {
    "n_data": mapper_batch_size,                                # The batch size 
    "num_iters": num_iters,                                     # The number of iterations of the prior optimization
    "lambd": (torch.tensor([0.5, 0.6, 0.6, 0.2])/D).to(device), # The regularization parameters for the layers/blocks
    "n_samples": 100,                                           # The number of function samples
    "lr": 5e-2,                                                 # The learning rate for the optimizer
    "print_every": 100,                                         # After how many epochs a evaluation should be printed
    "save_ckpt_every": 500                                      # After how many epochs a checkpoint should be saved
}

def prior_optimization(dataset_name, train_test_splits, dataset):
    out_dir = setup(dataset_name)
    n_splits = 10
    num_classes = 10
    masks_list = []
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id+1, dataset_name))
        # Load the dataset
        saved_dir = os.path.join(out_dir, str(split_id))
        split_data = train_test_splits[split_id]
        train_subset, _ = split_data
        train_indices = train_subset.indices

        X_train, y_train = extract_features_and_labels(train_indices, dataset)
        X_sample = dataset[train_indices[0]][0]

        input_dim, output_dim = int(X_sample.numel()), num_classes
        # Initialize data loader for the mapper
        data_loader = data_utils.DataLoader(train_subset, batch_size=mapper_batch_size, shuffle=True)
        # We draw measurement points from the training data
        rand_generator = ClassificationGenerator(data_loader)
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
        gp = GPR(X=X_train, Y=util.to_one_hot(y_train, num_classes),
                kern=kernel, mean_function=mean)
        # Initialize tunable MLP prior
        net_reparam = FactorizedGaussianPreResNetReparameterization(depth=20, D= D, prior_per="parameter",
                                                                    device = device)
        net_reparam = net_reparam.to(device)
        # Perform optimization
        mapper = PriorOptimisationMapper(out_dir=saved_dir, device=device, gp = gp, out_det = False).to(device)
        p_hist, loss_hist = mapper.optimize(net_reparam, rand_generator, output_dim=output_dim,
                                             **prior_opt_configurations)
        path = os.path.join(saved_dir, "loss_values.log")
        if not os.path.isfile(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)
        np.savetxt(path, loss_hist, fmt='%.6e')
        path = os.path.join(saved_dir, "pruned_values.log")
        if not os.path.isfile(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)
        np.savetxt(path, p_hist, fmt='%.6e')
        print("----" * 20)
        masks_list.append(net_reparam.get_det_masks())
    # Save the masks
    with open(os.path.join(out_dir, "masks_list.pkl"), "wb") as f:
        pickle.dump(masks_list, f)

# Configure the SGHMC sampler
sampling_configs = {
    "batch_size": 16,
    "num_samples": 50,
    "n_discarded": 0,
    "num_burn_in_steps": 5000,
    "keep_every": 5000,
    "lr": 1e-2,
    "num_chains": 2,
    "mdecay": 1e-2,
    "print_every_n_samples": 20
}
def train(dataset_name, train_test_splits, dataset):
    out_dir = setup(dataset_name)
    n_splits = 10
    with open(os.path.join(out_dir, "masks_list.pkl"), "rb") as f:
        masks_list = pickle.load(f)
    results = {"acc": [], "nll": []}
    for split_id in range(n_splits):
        print("Loading split {} of {} dataset".format(split_id, dataset_name))
        saved_dir = os.path.join(out_dir, str(split_id))
        
        # Load the dataset
        split_data = train_test_splits[split_id]
        train_subset, test_subset = split_data
        test_indices = test_subset.indices

        X_test, y_test = extract_features_and_labels(test_indices, dataset)
        # Initialize data loader for the mapper
        data_loader = data_utils.DataLoader(train_subset, batch_size=mapper_batch_size, shuffle=True)
        
        # Initialize the neural network and likelihood modules
        # Load the optimized prior for the MLP
        ckpt_path = os.path.join(out_dir, str(split_id), "ckpts", "it-{}.ckpt".format(num_iters))
        checkpoint = torch.load(ckpt_path)
        W_std_list = []
        b_std_list = []
        for key, value in checkpoint.items():
            if "W_std" in key:
                W_std_list.append(value)
            elif "b_std" in key:
                b_std_list.append(value)
        weight_masks, bias_masks, bias_mask_out = masks_list[split_id]
        net = MaskedPreResNet(depth=20, weight_masks=weight_masks, bias_masks=bias_masks, 
                              bias_mask_out=bias_mask_out, device = device)
        net = net.to(device)
        likelihood = LikCategorical()
        # Load the optimized prior for the classification net
        prior = OptimGaussianPrior(ckpt_path)
        # Initialize bayesian neural network with SGHMC sampler
        saved_dir = os.path.join(out_dir, str(split_id))
        bayes_net = ClassificationNet(net, likelihood, prior, saved_dir, n_gpu=0)
        
        # Start sampling
        bayes_net.sample_multi_chains(data_loader= data_loader, **sampling_configs)
        pred_mean, raw_preds = bayes_net.predict(X_test, True, True)
        p = raw_preds.cpu().numpy()
        p_mean = pred_mean.cpu().numpy()
        r_hat = compute_rhat_classification(p, sampling_configs["num_chains"])
        print("R-hat: mean {:.4f} std {:.4f}".format(float(r_hat.mean()), float(r_hat.std())))

        acc = uncertainty_metrics.accuracy(p_mean, y_test)
        acc = acc.cpu().item() if torch.is_tensor(acc) else acc
        nll = metrics_classification.nll(pred_mean, y_test)
        nll = nll.cpu().item() if torch.is_tensor(nll) else nll
        print("> Accuracy = {:.4f} | NLL = {:.4f}".format(acc, nll))
        data = [
            ["acc", "nll"],
            [acc, nll]
        ]
        # Save preliminary results in the saved_dir directory
        with open(os.path.join(saved_dir, "results.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
        results['acc'].append(acc)
        results['nll'].append(nll)

        result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(out_dir, "optim_results.csv"), sep="\t", index=False)

    print("Final results")
    print("> Accuracy: mean {:.4e} $\pm$ {:.4e} | NLL: mean {:.4e} $\pm$ {:.4e}".format(
            float(result_df['acc'].mean()), float(result_df['acc'].std()),
            float(result_df['nll'].mean()), float(result_df['nll'].std())))

def run(dataset_name):
    train_test_splits, dataset = data_splits(dataset_name)
    prior_optimization(dataset_name, train_test_splits, dataset)
    train(dataset_name, train_test_splits, dataset)

def run_training_only(dataset_name):
    train_test_splits, dataset = data_splits(dataset_name)
    train(dataset_name, train_test_splits, dataset)

def main():
    #fire.Fire(run)
    fire.Fire(run_training_only)

main()