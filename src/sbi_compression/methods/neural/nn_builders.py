import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_util as jtu
import equinox as eqx
import jax.linalg as jla
from jax import jit, vmap
from ..data import dataset_shapes, flatten_features   

def build_nn(input_shape: tuple,
             output_dim: int,
             hidden_dim: Union[int, tuple] = 128,
             activation: Callable = jnn.relu,
             use_cnn: bool = None,
             key: jax.random.PRNGKey = None,
             ) -> eqx.Module:
    """
    Build a user-defined neural network given specific input, hidden, and output dimensions.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input (excluding batch), e.g. (28, 28, 1) or (784,)
    output_dim : int
        Output dimension
    hidden_dim : list
        Hidden layer dimensions. 
    activation : callable
        Activation function (default: relu)
    key : jax.random.PRNGKey
        Random key for initialization
        
    Returns
    -------
    eqx.Module
        Initialized neural network (MLP or CNN)
    """
    
    NotImplementedError("This function is a placeholder and needs to be implemented.")

    
def build_2layer_nn(input_shape: tuple,
                    output_dim: int,
                    hidden_dim: Union[int, tuple] = 128,
                    activation: Callable = jax.nn.relu,
                    use_cnn: bool = None,
                    key: jax.random.PRNGKey = None,
                    ) -> eqx.Module:
    """
    Build a 2-hidden-layer network adapted to input shape.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input (excluding batch), e.g. (28, 28, 1) or (784,)
    output_dim : int
        Output dimension
    hidden_dim : int or tuple
        Hidden layer dimensions. If int, use same for both layers.
        If tuple, (hidden1_dim, hidden2_dim).
    activation : callable
        Activation function (default: relu)
    use_cnn : bool, optional
        Force CNN (True) or MLP (False). If None, auto-detect:
        - CNN if input is 3D or higher (spatial data)
        - MLP if input is 1D or 2D
    key : jax.random.PRNGKey
        Random key for initialization
        
    Returns
    -------
    eqx.Module
        Initialized neural network (MLP or CNN)
    """
    NotImplementedError("This function is a placeholder and needs to be implemented.")


def _2layer_hidden_dims(input_shape: tuple, 
                       output_dim: int
                       ) -> tuple:
    """
    Utility function to determine hidden layer dimensions for a 2-hidden-layer network.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input (excluding batch), e.g. (28, 28, 1) or (784,)
    output_dim : int
        Output dimension
    hidden_dim : int or tuple
        Hidden layer dimensions. If int, use same for both layers.
        If tuple, (hidden1_dim, hidden2_dim).
        
    Returns
    -------
    tuple
        Tuple of hidden layer dimensions (hidden1_dim, hidden2_dim)
    """
    hidden_dims = []
    if len(input_shape) == 1:
        type = "MLP"
        increment = int((input_shape[0] - output_dim)/3)
        if increment <= 0:
            raise ValueError("Input dimension is too small compared to output dimension. Please choose a smaller output dimension or provide explicit hidden dimensions.")
        hidden_dim1 = input_shape[0] - increment
        hidden_dim2 = input_shape[0] - 2*increment
        hidden_dims.append(hidden_dim1)
        hidden_dims.append(hidden_dim2)
    elif len(input_shape) == 2 or len(input_shape) == 3:
        type = "CNN"
        increment = int((input_shape[0] - output_dim)/3)
        flat_input_dim = jnp.prod(jnp.asarray(input_shape))
        hidden_dim1 = (jnp.sqrt(flat_input_dim))).astype(int)
        if hidden_dim1[0] < output_dim:
            hidden_dim1[0] = max(hidden_dim1[0], output_dim)
        if hidden_dim1[1] < 1:
            hidden_dim1[1] = max(hidden_dim1[1], 1)
        hidden_dim2 = jtu.tree_map(lambda x: x - 2*increment, input_shape[:1])
        if hidden_dim2[0] < output_dim:
            hidden_dim2 = jtu.tree_map(lambda x: max(x, output_dim), hidden_dim2)
        hidden_dims.append(hidden_dim1)
        hidden_dims.append(hidden_dim2)

    else:
        raise ValueError("Input shape should be 1D, 2D, or 3D (excluding batch dimension).")
    

