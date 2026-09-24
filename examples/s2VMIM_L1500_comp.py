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
import ast
from sbi_compression.methods.neural.flows import RQSplineFlow
import distrax

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

    # @nnx.remat
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

class s2_AE_Flow(nnx.Module):
    def __init__(self,
        encoder: nnx.Module,
        flow: nnx.Module,
        encoder_mode: Literal['context', 'features'],
        verbose: bool = False,
        ):
        self.encoder_mode = encoder_mode
        self.encoder = encoder
        self.encoder.verbose = verbose
        self.flow = flow
        self.verbose = verbose
    
    def __call__(self, x: Array, context: Array) -> Array:
        if self.encoder_mode == 'context':
            context_encoded = self.encoder(context)
            x = self.flow(x, context_encoded)
        elif self.encoder_mode == 'features':
            x_encoded = self.encoder(x)
            x = self.flow(x_encoded, context)
        jax.debug.print("Past flow.") if self.verbose else None
        return x
    
    def sample(self, num_samples: int, rng: Array, context: Array) -> Array:
        x = self.flow.sample(num_samples, rng, context)
        return x
        
    def encode(self, x: Array) -> Array:
        # assert x.shape[-len(self.input_shape):] == self.input_shape # Make sure the input shape matches the encoder, assume batch first
        x = self.encoder(x)
        return x
    
    def mode(self, mode: Literal['train_encoder', 'train_flow', 'train_all', 'eval']) -> None:
        if mode == 'train_encoder':
            self.encoder.train()
            self.flow.eval()
        elif mode == 'train_flow':
            self.encoder.dropout = False
            self.encoder.eval()
            self.flow.train()
        elif mode == 'train_all':
            self.encoder.train()
            self.flow.train()
        elif mode == 'eval':
            self.encoder.dropout = False
            self.encoder.eval()
            self.flow.eval()

def nll_loss(model, x, context):
    loss = -jnp.mean(model(x, context))
    return loss

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

@nnx.jit
def encode_step(model, x):
    """Run the encoder under jit so XLA can partition it across the sharded batch."""
    return model.encode(x)



# ------ Compress ------
def compress() -> None:

    parser = argparse.ArgumentParser(description='Train a spherical CNN MSE compression model.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--train_test_split', '-tts', type=float, default=0.8, help='Train test split.')
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size.')
    parser.add_argument('--steps', '-s', type=int, default=2200, help='Number of steps.')
    parser.add_argument('--print_every', type=int, default=50, help='Print every n steps.')
    parser.add_argument('--L', '-L', type=int, default=750, help='Spherical harmonic cutoff.')
    parser.add_argument('--N', '-N', type=int, default=6000, help='Number of maps.')
    parser.add_argument('--n_samples', type=int, default=6000, help='Number of compressed datapoints.')
    parser.add_argument('--data_directory', '-data_dir', type=str, 
        # default='/scratch/u6pf/brianycc/spherical_maps/5_tomo_bins_flm_{L}_all_mwss/', 
        default='/projects/u6pf/brianycc/spherical_maps/map_compression_L1500_mwss_samples_augmented_pooled',
        help='Directory containing the maps. Default takes L from the command line.'
        )
    parser.add_argument('--output_directory', '-output_dir', type=str, 
        # default='/scratch/u6pf/brianycc/spherical_maps/5_tomo_bins_flm_{L}_all_mwss/', 
        default='/projects/u6pf/brianycc/spherical_maps/map_compression_L1500_mwss_samples_augmented_pooled_compressed',
        help='Directory to save the compressed maps to. Default takes L from the command line.'
        )
    parser.add_argument('--normalise_params', action='store_true', help='Normalise the parameters.')
    parser.add_argument('--normalise_maps', action='store_true', help='Normalise the maps.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode.')
    parser.add_argument('--dropout', action='store_true', help='Dropout mode.')
    parser.add_argument('--save_compressed', action='store_true', help='Save the model.')
    parser.add_argument('--n_transforms', type=int, default=4, help='Number of transforms in flow model.')
    parser.add_argument('--n_bins', type=int, default=8, help='Number of bins in the spline function.')
    parser.add_argument('--conditioner_dimensions', type=ast.literal_eval, 
        default='((32,32), (32,32))', 
        help="Dimensions of the MLP conditioner for RQSpline. Tuple structure, e.g., '((32,32), (32,32))'"
        )

    args = parser.parse_args()
    LEARNING_RATE = args.learning_rate
    TRAIN_TEST_SPLIT = args.train_test_split
    BATCH_SIZE = args.batch_size
    STEPS = args.steps
    PRINT_EVERY = args.print_every
    L = args.L
    N = args.N
    n_samples = args.n_samples
    DATA_DIR = args.data_directory.format(L=L, N=N)
    OUTPUT_DIR = args.output_directory
    NORM_PARAMS = args.normalise_params
    NORM_MAPS = args.normalise_maps
    VERBOSE = args.verbose
    DROPOUT = args.dropout
    save_compressed = args.save_compressed

    import pickle

    # --- LOADING ---
    # 1. Create a fresh instance of the model with the SAME hyperparameters
    n_transforms = args.n_transforms
    n_bins = args.n_bins

    name = l = f's2VMIM_L{L}_{N}N_{STEPS}steps_{BATCH_SIZE}batch_{LEARNING_RATE}lr_{n_transforms}transforms_{n_bins}bins'
    s2_CNN = s2CNN()

    from sbi_compression.methods.neural.flows import RQSplineFlow
    import distrax

    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    devices = jax.devices()
    mesh = Mesh(devices, ('batch',))
    sharding_data = NamedSharding(mesh, P('batch', None)) # For inputs'

    features_shape = (5,)
    context_shape = (5,)
    features_dim = prod(features_shape)
    context_dim = prod(context_shape)
    n_transforms = args.n_transforms
    n_bins = args.n_bins
    range_min = -4
    range_max = 4
    bijector_type = distrax.RationalQuadraticSpline
    conditioner_hidden_dims = args.conditioner_dimensions
    activation = "gelu"
    flow = RQSplineFlow(
        features_dim, 
        context_dim, 
        n_transforms=n_transforms, 
        hidden_dims=conditioner_hidden_dims, 
        activation=getattr(nnx, activation), 
        n_bins=n_bins, 
        range_min=range_min, 
        range_max=range_max, 
        bijector_type=bijector_type
        )
        
    load_model = s2_AE_Flow(s2_CNN, flow, 'context', verbose=False)

    # 2. Load the state from the file
    with open(f'examples/checkpoints/{name}.pkl', 'rb') as f:
        loaded_state = pickle.load(f)
        print(f"Loaded model: {name} ")

    # 3. Update the model with the loaded state
    nnx.update(load_model, loaded_state)

    # 4. Replicate the model weights across every device in the mesh so the
    #    jitted encode step can pair them with the batch-sharded input below.
    replicated_sharding = NamedSharding(mesh, P())
    model_state = nnx.state(load_model, nnx.Param)
    model_state = jax.device_put(model_state, replicated_sharding)
    nnx.update(load_model, model_state)

    directory = args.data_directory

    gt = ground_truth = []
    cd = compressed_data = []

    dataset = SphericalMapDataset(
        # directory="/projects/u6pf/brianycc/spherical_maps/map_compression_samples_5_tomo_bins_flm_1500_batch_1/", 
        directory = directory, 
        n_maps = n_samples, 
        param_filename = "params.npy",
        normalise_params = NORM_PARAMS,
        normalise_maps = NORM_MAPS
        )
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)
    for x_batch, p_batch in eval_loader:
        x_jax = jnp.asarray(x_batch.detach().numpy() if hasattr(x_batch, 'detach') else x_batch)
        p_jax = jnp.asarray(p_batch.detach().numpy() if hasattr(p_batch, 'detach') else p_batch)
        x_sharded = jax.device_put(x_jax, sharding_data)
        p_sharded = jax.device_put(p_jax, sharding_data)
        cd.append(encode_step(load_model, x_sharded))
        gt.append(jnp.asarray(p_sharded))
    cd = np.concatenate([np.array(d) for d in cd])
    gt = np.concatenate([np.array(t) for t in gt])
    compressed_dataset = {
        "ground_truth": gt,
        "compressed_data": cd,
    }
    print("Compressed data shape", cd.shape, "\nGround truth shape", gt.shape)
    if save_compressed:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(f"{OUTPUT_DIR}/{name}.pkl", "wb") as f:
            pickle.dump(compressed_dataset, f)
        print(f"Saved compressed data and ground truth to \n{OUTPUT_DIR}/{name}_{n_samples}samples.pkl")

    params = ['Oc','Ob','h','As','n_s']
    num_params = gt.shape[1]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10,6))
    axes = axes.flatten()
    for i in range(num_params):
        ax = axes[i]
        ax.scatter(gt[:,i], cd[:,i], s=1)
        ax.set_xlabel(f'True {params[i]}')
        ax.set_ylabel(f'Latent feature {i}')

    plt.suptitle("s2VMIM compressor")
    plt.tight_layout()
    plt.savefig(f"examples/plots/s2VMIM_L{L}_scatter_{STEPS}steps.pdf")
    plt.show()


if __name__ == '__main__':
    compress()

































































































































































