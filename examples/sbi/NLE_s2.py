# general 
import scipy as sp
import pickle
from tqdm import tqdm
import json
import getdist
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
import numpy as np
import argparse
import ast
import os

# density estimation 
from sbi.inference import NPE, NLE
import torch
from sbi.analysis import pairplot
from sbi.analysis import plot_summary
from sbi.utils import BoxUniform
from sbi.diagnostics import run_sbc
from sbi.analysis.plot import sbc_rank_plot

import subprocess
print("--- PyTorch & CUDA Status ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Compiled CUDA Version: {torch.version.cuda}")
print("\n--- System GPU Driver Status ---")
try:
    smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    # Print the top lines showing Driver Version and CUDA Version
    for line in smi_output.split('\n')[:4]:
        print(line)
except Exception as e:
    print("Could not run nvidia-smi:", e)

import multiprocessing
num_cores = multiprocessing.cpu_count()
print(f'Number of CPU cores available: {num_cores}')

if torch.cuda.is_available():
    device = 'cuda'
    print(f'Device: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    device = 'mps'
    print(f'Device: MPS')    
else:
    device = 'cpu'
    print('Device: CPU')


def train_and_sample_posterior() -> None:

    parser = argparse.ArgumentParser(description='Train an SBI model and save both the model and samples.')
    parser.add_argument('--compression_types', type=ast.literal_eval, 
        help="Name of the copmression used for the data used for SBI."
        )
    parser.add_argument('--compression_directories', type=ast.literal_eval, 
        help="Directory of the compressed data used for training SBI model."
        )
    parser.add_argument('--n_samples', type=int, default=3000, help='Number of posterior samples.')
    
    args = parser.parse_args()
    compression_types = args.compression_types
    compression_directories = args.compression_directories
    n_samples = args.n_samples

    with open(compression_directories, 'rb') as f:
        load = pickle.load(f)
        gt = load["ground_truth"]
        cd = load["compressed_data"]
        print(f'The dimension of the parameter datset is', gt.shape)
        print(f'The dimension of the compressed data dataset is', cd.shape)
    N = np.array(gt).shape[0]
    print(f'There are {N} datapoints.')
    gt = torch.tensor(gt, dtype=torch.float32)
    cd = torch.tensor(cd, dtype=torch.float32)
    # Check whether the trainig data is of correct shape and network has been instantiated
    print('parameter samples', type(gt), gt.shape, 'cls samples', type(cd), cd.shape)
    
    nle = NLE(density_estimator="nsf", device=device)
    print('Density estimation model:', nle)
    nle = nle.append_simulations(gt, cd, data_device=device)
    nle.train(
        training_batch_size=64,
        learning_rate=1e-3,
    )
    _ = plot_summary(nle)
    os.makedirs("examples/trained_sbi_models", exist_ok=True)
    with open(f"examples/trained_sbi_models/NLE_{compression_types}_{N}N.pkl", "wb") as f:
        pickle.dump(nle, f)
    print("Completed training of NLE model")

    prior = BoxUniform(
        low=torch.tensor([0.05, 0.01875, 0.64, 1.61, 0.84], device=device),
        high=torch.tensor([0.255, 0.02625, 0.82, 3.91, 1.1], device=device)
    )

    posterior = nle.build_posterior(prior=prior)
    print(posterior)
    cd_obs = cd[42, :].clone().detach().to(device)
    gt_obs = gt[42, :].clone().detach().to(device)
    print('The compressed observation is', cd_obs)
    print('The ground truth is', gt_obs)
    samples = posterior.sample((n_samples,), x=cd_obs).cpu().numpy()
    posterior_data = {
        "samples": samples,
        "observation": cd_obs,
        "ground_truth": gt_obs,
    }

    os.makedirs("examples/posterior_samples", exist_ok=True)
    with open(f"examples/posterior_samples/NLE_{compression_types}_posterior_{n_samples}samples.pkl", "wb") as f:
        pickle.dump(posterior_data, f)
        print(f"posterior samples saved at: \nexamples/posterior_samples/NLE_{compression_types}_posterior_{n_samples}samples.pkl")


if __name__ == '__main__':
    train_and_sample_posterior()