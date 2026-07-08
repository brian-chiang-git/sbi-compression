from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm


@nnx.jit(static_argnames="loss_fn")
def train_step(model, optimizer: nnx.Optimizer, loss_fn, x_batch, context_batch):
    """Train for a single step."""
    loss_value, grads = nnx.value_and_grad(loss_fn)(model, x_batch, context_batch)
    optimizer.update(model, grads)  # In-place updates.
    return loss_value

@nnx.jit(static_argnames="loss_fn")
def eval_step(model, loss_fn, x, y):
    """Calculate loss on test data without updating parameters."""
    loss_value = loss_fn(model, x, y)
    return loss_value

def train_model(model, 
                x: Array, 
                p: Optional[Array] = None, 
                optimizer: nnx.Optimizer, 
                loss_fn: Callable, 
                train_test_split: float,
                batch_size: int,
                n_steps: int, 
                eval_freq: int,
                shuffle: bool = True,
                drop_last: bool = True,
                ):
    if p is not None:
        assert x.shape[0] == p.shape[0], "x and p must have the same number of samples"
        x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
        p_tensor = torch.tensor(np.array(p), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, p_tensor)
    else:
        x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    def infinite_trainloader():
        while True:
            yield from train_loader

    train_losses = []
    test_losses = []
    test_steps = []
    if p is not None:
        for step, (x_batch, p_batch) in tqdm(zip(range(n_steps), infinite_trainloader())):
            train_loss = train_step(model, optimizer, loss_fn, p_batch.numpy(), x_batch.numpy()) # Posteriior Estimation
            train_losses.append(train_loss)
            # --- EVALUATION PHASE ---
            if step % eval_freq == 0:
                # metrics.reset() # Clear training metrics to track test metrics
                test_loss = 0
                for batch_x, batch_p in test_loader:
                    test_loss += eval_step(model, loss_fn, batch_p.numpy(), batch_x.numpy())
                test_loss /= len(test_loader)
                test_losses.append(test_loss)
                test_steps.append(step)
                print(f"Step {step:3d} ({(step*batch_size)/train_size:.1f} epoch) | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    else:
        for step, (x_batch) in tqdm(zip(range(n_steps), infinite_trainloader())):
            train_loss = train_step(model, optimizer, loss_fn, x_batch.numpy()) # Posteriior Estimation
            train_losses.append(train_loss)
            # --- EVALUATION PHASE ---
            if step % eval_freq == 0:
                # metrics.reset() # Clear training metrics to track test metrics
                test_loss = 0
                for batch_x in test_loader:
                    test_loss += eval_step(model, loss_fn, batch_x.numpy())
                test_loss /= len(test_loader)
                test_losses.append(test_loss)
                test_steps.append(step)
                print(f"Step {step:3d} ({(step*BATCH_SIZE)/train_size:.1f} epoch) | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
    print("Training completed.")


import sys
import types
import importlib

def universal_flax_nnx_shim():
    """
    Ensures absolute cross-compatibility between legacy (experimental/0.10) 
    and modern (0.12+) Flax NNX architectures at runtime.
    """
    try:
        # 1. Inspect the currently installed Flax version
        flax_spec = importlib.util.find_spec("flax")
        if flax_spec is None:
            print("Flax is not installed in this environment. Skipping shim.")
            return
            
        flax = importlib.import_module("flax")
        flax_version = getattr(flax, "__version__", "0.0.0")
        major_minor = tuple(map(int, flax_version.split(".")[:2]))

        # 2. If modern Flax (>= 0.11), map old paths to new objects
        if major_minor >= (0, 11):
            if not hasattr(flax, "nnx"):
                print("Modern Flax detected but 'nnx' module missing.")
                return
                
            nnx = flax.nnx
            
            # Create a virtual 'flax.nnx.nnx.structures' module
            if "flax.nnx.nnx.structures" not in sys.modules:
                mock_structures = types.ModuleType("flax.nnx.nnx.structures")
                mock_structures.List = getattr(nnx, "List", list)
                mock_structures.Dict = getattr(nnx, "Dict", dict)
                sys.modules["flax.nnx.nnx.structures"] = mock_structures
            
            # Map top level 'flax.nnx.nnx' fallback
            if "flax.nnx.nnx" not in sys.modules:
                sys.modules["flax.nnx.nnx"] = nnx
                
            # Create a virtual 'flax.experimental.nnx.nnx.structures' module
            if "flax.experimental.nnx.nnx.structures" not in sys.modules:
                mock_exp = types.ModuleType("flax.experimental.nnx.nnx.structures")
                mock_exp.List = getattr(nnx, "List", list)
                mock_exp.Dict = getattr(nnx, "Dict", dict)
                sys.modules["flax.experimental.nnx.nnx.structures"] = mock_exp
                
            print(f"Modern Flax ({flax_version}) detected: Legacy NNX shim applied successfully.")

        # 3. If legacy Flax (< 0.11), map new paths to old objects
        else:
            # Check if using the older 0.10 native setup or the 0.8 experimental setup
            try:
                from flax.nnx.nnx import structures as legacy_structures
                nnx_module = importlib.import_module("flax.nnx.nnx")
            except ImportError:
                try:
                    from flax.experimental.nnx.nnx import structures as legacy_structures
                    nnx_module = importlib.import_module("flax.experimental.nnx.nnx")
                except ImportError:
                    print("⚠️ Older Flax detected but internal NNX components could not be resolved.")
                    return
            
            # Expose 'flax.nnx' globally so modern code calling `from flax import nnx` works seamlessly
            if not hasattr(flax, "nnx"):
                setattr(flax, "nnx", nnx_module)
                sys.modules["flax.nnx"] = nnx_module
                
            print(f"🛰️ Legacy Flax ({flax_version}) detected: Modern NNX forwarding shim applied successfully.")

    except Exception as e:
        print(f"❌ Critical failure applying universal Flax NNX shim: {e}")