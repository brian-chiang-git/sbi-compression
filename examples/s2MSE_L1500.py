import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TMPDIR"] = "/tmp"
os.environ["TMP"] = "/tmp"
os.environ["TEMP"] = "/tmp"
print("--- Environment Variables inside Container ---")
print("CUDA_VISIBLE_DEVICES  :", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("\n--- Testing nvidia-smi ---")
os.system("nvidia-smi")

import socket
print("=== VS CODE CONTAINER DIAGNOSTIC ===")
print("1. Hostname               :", socket.gethostname())
print("2. Slurm Job ID           :", os.environ.get("SLURM_JOB_ID", "NONE (Running outside Slurm!)"))
print("3. CUDA_VISIBLE_DEVICES   :", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("4. Apptainer GPU Devices  :", [f for f in os.listdir('/dev') if 'nvidia' in f] if os.path.exists('/dev') else "No /dev")

import sys
print("1. Python Binary  :", sys.executable)
print("2. Site-Packages  :", [p for p in sys.path if "venv" in p])
print("3. CUDA Available :")

import torch
print("torch", torch.__version__, torch.cuda.is_available(), f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")

import jax
jax.config.update("jax_enable_x64", True)
print("Jax version", jax.__version__, "64-bit", jax.config.read("jax_enable_x64"))
print("jax", jax.__version__, "GPU backend:", jax.default_backend() == 'gpu', f"({jax.devices('gpu')[0].device_kind})" if jax.default_backend() == 'gpu' else "")

import dataclasses
import numpy as np
import jax.numpy as jnp
import flax
print("Flax version", flax.__version__)
from flax import nnx
import optax
from torch.utils.data import Dataset, DataLoader, random_split
from s2ai.blocks.core_blocks import DiscoConvBlock, AverageBlock
from typing import Any, List, Optional, Callable, Sequence, Tuple, Union
from jaxtyping import Array, Float, Int, PyTree # https://github.com/google/jaxtyping
from typing import Literal, Callable, Optional
from tqdm import tqdm
from math import prod
import s2fft
import pickle
import argparse

class SphericalMapDataset(Dataset):
    def __init__(self, 
        directory, 
        n_maps, 
        param_filename = "params.npy",
        stats_filename="map_stats.npz",
        normalise_params = True,
        normalise_maps = True,
        ):
        """
        Args:
            directory (str): Path to the folder containing map_{i}.npy.
            n_maps (int): Total number of map samples to load.
            param_filename (str): Name of the parameters numpy file.
        """
        self.directory = directory
        self.n_maps = n_maps
        self.normalise_maps = normalise_maps
        self.normalise_params = normalise_params

        # The parameters file is tiny, so it is safe to load entirely into RAM
        param_path = os.path.join(directory, param_filename)
        raw_params = np.load(param_path)
        assert self.n_maps <= len(raw_params), f"Requested {self.n_maps} maps, but params.npy only has {len(raw_params)} rows!"
        
        # --- 1. Parameter Standardization ---
        self.normalise_params = normalise_params
        if self.normalise_params:
            self.param_mean = np.mean(raw_params, axis=0, keepdims=True)
            self.param_std = np.std(raw_params, axis=0, keepdims=True) #+ 1e-8
            self.params = (raw_params - self.param_mean) / self.param_std
        else:
            self.params = raw_params
            # self.param_mean = np.zeros((1, raw_params.shape[1]))
            # self.param_std = np.ones((1, raw_params.shape[1]))

        # 2. Load Pre-computed Map Stats
        if self.normalise_maps:
            stats_path = os.path.join(directory, stats_filename)
            assert os.path.exists(stats_path), f"Stats file {stats_path} not found! Run compute_and_save_map_stats first."
            
            stats = np.load(stats_path)
            # Shape for broadcasting over (theta, phi, channels): (1, 1, channels)
            self.map_mean = stats["mean"].reshape(1, 1, -1)
            self.map_std = stats["std"].reshape(1, 1, -1) #+ 1e-8

    def __len__(self):
        return self.n_maps
    
    def __getitem__(self, idx):
        # Lazily load only ONE map to memory on demand
        map_path = os.path.join(self.directory, f"map_{idx}.npy")
        m = np.load(map_path)
        p = self.params[idx].astype(np.float32)
        # Standardize Map dynamically per channel
        if self.normalise_maps:
            m = (m - self.map_mean) / self.map_std
        return m, p

class RectilinearAverageBlock(nnx.Module):
    def __init__(self):
        pass  # No spherical weights needed anymore

    def __call__(self, x):
        # x shape: (batch, theta, phi, channels) -> e.g., (1, 751, 1500, 5)
        # Average over both spatial dimensions (theta and phi) while keeping batch and channels
        return jnp.mean(x, axis=(1, 2))

class s2CNN(nnx.Module):
    """Disco convolutional network with spherical group normalisation."""

    def __init__(
        self,
        # modes: List[str] = dataclasses.field(
        #     default_factory=lambda: ["disco", "disco", "disco", "disco"]
        # ),
        modes: tuple[str] = ("disco", "disco", "disco", "disco"),
        rngs: nnx.Rngs = nnx.Rngs(0),
        verbose: bool = True,
        dropout: bool = True,
    ):
        self.verbose = verbose
        self.dropout = dropout

        # These could be nnx.vmapped/scanned for faster initialization
        self.disco_conv_block_0 = DiscoConvBlock(
            (1500, 750), (5, 10), 10, mode=modes[0], rngs=rngs
        )
        self.disco_conv_block_1 = DiscoConvBlock(
            (750, 250), (10, 20), 20, mode=modes[1], rngs=rngs
        )
        self.disco_conv_block_2 = DiscoConvBlock(
            (250, 50), (20, 40), 40, mode=modes[2], rngs=rngs
        )
        self.disco_conv_block_3 = DiscoConvBlock(
            (50, 10), (40, 80), 80, mode=modes[3], rngs=rngs
        )

        # self.average_block = AverageBlock(L_in=10)
        self.average_block = RectilinearAverageBlock()

        self.dropout_4 = nnx.Dropout(0.5, rngs=rngs)
        self.dense_4 = nnx.Linear(
            in_features=160,
            out_features=80,
            rngs=rngs,
        )
        self.activation_4 = nnx.gelu

        self.dropout_5 = nnx.Dropout(0.5, rngs=rngs)
        self.dense_5 = nnx.Linear(
            in_features=80,
            out_features=40,
            rngs=rngs,
        )
        self.activation_5 = nnx.gelu

        self.dropout_6 = nnx.Dropout(0.5, rngs=rngs)
        self.dense_6 = nnx.Linear(
                    in_features=40,
                    out_features=20,
                    rngs=rngs,
                )
        self.activation_6 = nnx.gelu

        self.dropout_7 = nnx.Dropout(0.5, rngs=rngs)
        self.dense_7 = nnx.Linear(
            in_features=20,
            out_features=5,
            rngs=rngs,
        )

    @nnx.remat
    def __call__(self, x):
        # Spherical convolutional blocks (L's, channels, groups)
        x = self.disco_conv_block_0(x)
        x = self.disco_conv_block_1(x)
        x = self.disco_conv_block_2(x)
        x = self.disco_conv_block_3(x)
        jax.debug.print("Past disco block.") if self.verbose else None
        # Convert to equivariant dense layer
        x = self.average_block(x)
        # Standard MLP classifier
        # x = self.dropout_4(x) if self.dropout else x
        # x = self.dense_4(x)
        # x = self.activation_4(x)
        x = self.dropout_5(x) if self.dropout else x
        x = self.dense_5(x)
        x = self.activation_5(x)
        x = self.dropout_6(x) if self.dropout else x
        x = self.dense_6(x)
        x = self.activation_6(x)
        x = self.dropout_7(x) if self.dropout else x
        x = self.dense_7(x)
        jax.debug.print("Past dense mlp.") if self.verbose else None
        return x

def mse_loss(model, x, y):
    preds = model(x)
    if preds.shape != y.shape:
        raise ValueError(f"Output shpae of the model {preds.shape} does not match the shape of the labels {y.shape}")
    return jnp.mean((preds-y) ** 2)

@nnx.jit(static_argnames="loss_fn")
def train_step(model, optimizer: nnx.Optimizer, loss_fn, x_batch, y_batch):
    """Train for a single step."""
    loss_value, grads = nnx.value_and_grad(loss_fn)(model, x_batch, y_batch)
    if hasattr(model, 'verbose') and model.verbose:
        jax.debug.print("Dense 7 grads norm: {}", jax.tree_util.tree_map(jnp.linalg.norm, grads.dense_7))
    optimizer.update(model, grads)  # In-place updates.
    return loss_value

@nnx.jit(static_argnames="loss_fn")
def eval_step(model, loss_fn, x, context):
    """Calculate loss on test data without updating parameters."""
    loss_value = loss_fn(model, x, context)
    return loss_value




# ------ Training ------
def train_model() -> None:

    parser = argparse.ArgumentParser(description='Train a spherical CNN MSE compression model.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--train_test_split', '-tts', type=float, default=0.8, help='Train test split.')
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size.')
    parser.add_argument('--steps', '-s', type=int, default=2200, help='Number of steps.')
    parser.add_argument('--print_every', type=int, default=50, help='Print every n steps.')
    parser.add_argument('--L', '-L', type=int, default=750, help='Spherical harmonic cutoff.')
    parser.add_argument('--N', '-N', type=int, default=6000, help='Number of maps.')
    parser.add_argument('--data_directory', '-data_dir', type=str, 
        # default='/scratch/u6pf/brianycc/spherical_maps/5_tomo_bins_flm_{L}_all_mwss/', 
        default='/projects/u6pf/brianycc/spherical_maps/map_compression_L1500_mwss_samples_augmented_pooled',
        help='Directory containing the maps. Default takes L from the command line.'
        )
    parser.add_argument('--normalise_params', action='store_true', help='Normalise the parameters.')
    parser.add_argument('--normalise_maps', action='store_true', help='Normalise the maps.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode.')
    parser.add_argument('--dropout', action='store_true', help='Dropout mode.')
    parser.add_argument('--save_model', action='store_true', help='Save the model.')
    
    args = parser.parse_args()
    LEARNING_RATE = args.learning_rate
    TRAIN_TEST_SPLIT = args.train_test_split
    BATCH_SIZE = args.batch_size
    STEPS = args.steps
    PRINT_EVERY = args.print_every
    L = args.L
    N = args.N
    DATA_DIR = args.data_directory.format(L=L, N=N)
    NORM_PARAMS = args.normalise_params
    NORM_MAPS = args.normalise_maps
    VERBOSE = args.verbose
    DROPOUT = args.dropout
    SAVE_MODEL = args.save_model

    # sharding to utilise all gpus
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    devices = jax.devices()
    mesh = Mesh(devices, ('batch',))
    sharding_data = NamedSharding(mesh, P('batch', None)) # For inputs

    S2_MSE = s2CNN(verbose=False)

    optimizer = nnx.Optimizer(
        S2_MSE, 
        optax.adamw(LEARNING_RATE),
        wrt=nnx.Param
    )

    train_losses = []
    test_losses = []
    test_steps = []

    dataset = SphericalMapDataset(
        directory=DATA_DIR, 
        n_maps=N, 
        param_filename="params.npy",
        normalise_params=NORM_PARAMS,
        normalise_maps=NORM_MAPS
        )
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, 
        num_workers=4, 
        persistent_workers=True
        )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, 
        num_workers=4, 
        persistent_workers=True
        )

    def infinite_trainloader():
        while True:
            yield from train_loader

    for step, (x_batch, p_batch) in tqdm(zip(range(STEPS), infinite_trainloader())):
        S2_MSE.dropout = True
        x_jax = jnp.asarray(x_batch.detach().numpy() if hasattr(x_batch, 'detach') else x_batch)
        p_jax = jnp.asarray(p_batch.detach().numpy() if hasattr(p_batch, 'detach') else p_batch)
        x_sharded = jax.device_put(x_jax, sharding_data)
        p_sharded = jax.device_put(p_jax, sharding_data)
        train_loss = train_step(S2_MSE, optimizer, mse_loss, x_sharded, p_sharded) # Posteriior Estimation
        # Block until GPU finishes step (prevents asynchronous memory buildup)
        # train_loss.block_until_ready()
        train_losses.append(float(train_loss))
        # Explicitly delete batch references so Python garbage collector frees host RAM
        del x_batch, p_batch, x_jax, p_jax, x_sharded, p_sharded
        # --- EVALUATION PHASE ---
        if step % PRINT_EVERY == 0:
            # metrics.reset() # Clear training metrics to track test metrics
            S2_MSE.dropout = False
            test_loss = 0
            for batch_x, batch_p in test_loader:
                jax_x = jnp.asarray(batch_x.detach().numpy() if hasattr(batch_x, 'detach') else batch_x)
                jax_p = jnp.asarray(batch_p.detach().numpy() if hasattr(batch_p, 'detach') else batch_p)
                sharded_x = jax.device_put(jax_x, sharding_data)
                sharded_p = jax.device_put(jax_p, sharding_data)
                test_loss += eval_step(S2_MSE, mse_loss, sharded_x, sharded_p)
            test_loss /= len(test_loader)
            # test_loss.block_until_ready()
            test_losses.append(float(test_loss))
            del batch_x, batch_p, jax_x, jax_p, sharded_x, sharded_p
            test_steps.append(step)
            print(f"Step {step:3d} ({(step*BATCH_SIZE)/train_size:.1f} epoch) | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    print("Training completed.")

    if SAVE_MODEL:
        state = nnx.state(S2_MSE, nnx.Param)
        name = f's2MSE_L{L}_{N}N_{STEPS}steps_{BATCH_SIZE}batch_{LEARNING_RATE}lr'
        os.makedirs('examples/checkpoints', exist_ok=True)
        with open(f'examples/checkpoints/{name}.pkl', 'wb') as f:
            pickle.dump(state, f)
            print("Model saved at: ", f'\nexamples/checkpoints/{name}.pkl')
        history = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_steps': test_steps
        }
        with open(f'examples/checkpoints/{name}_history.pkl', 'wb') as f:
            pickle.dump(history, f)
            print(f"Loss history saved at: examples/checkpoints/{name}_history.pkl")


if __name__ == '__main__':
    train_model()

































































































































































