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