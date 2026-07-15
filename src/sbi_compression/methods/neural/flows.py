from typing import Any, List, Optional, Callable, Sequence, Tuple, Union
from jaxtyping import Array, Float, Int, PyTree # https://github.com/google/jaxtyping
from jax.random import PRNGKey

import jax
import jax.numpy as jnp
import numpy as np
import math
import distrax
import flax
from flax import nnx
# from sbi_compression.methods.utils import universal_flax_nnx_shim
# universal_flax_nnx_shim()

from .bijectors import ConditionalChain, ConditionalMaskedCoupling, ConditionalInverse, ConditionalTransformed 
from .nn_nnx import MLP     

class Conditioner(nnx.Module):
    def __init__(self,
                 features_shape: tuple,
                 context_shape: tuple,
                 hidden_dims: Sequence[tuple],
                 num_bijector_params: int,
                 activation: Callable = nnx.leaky_relu,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 kernel_init: Callable = nnx.initializers.lecun_normal(),
                 ):
        self.features_shape = features_shape
        self.context_shape = context_shape
        self.num_bijector_params = num_bijector_params
        self.activation = activation

        self.n_flat_features = jax.tree.reduce(lambda carry, x: carry*x, features_shape)
        self.n_flat_context = jax.tree.reduce(lambda carry, x: carry*x, context_shape)
        self.layer_dims = ((self.n_flat_features+self.n_flat_context, hidden_dims[0][0]),) + hidden_dims

        self._conditioner = nnx.List()
        self._conditioner_mlp = MLP(self.layer_dims, activation, rngs=rngs, kernel_init=kernel_init) # Build the NN as specified by layer_dims that learns the sline transformation parameters
        self._conditioner_out = nnx.Linear(self.layer_dims[-1][-1],
                                            self.n_flat_features*num_bijector_params,
                                            rngs=rngs,
                                            kernel_init=kernel_init
                                            )
    
    def __call__(self, x, context=None):
        # Flatten feature vector to prepare for parsing into spline transform Conditioner, which is a multilayer perceptron
        x_batch_shape = x.shape[:-len(self.features_shape)]
        x = x.reshape(*x_batch_shape, -1)

        if context is not None:
            # Stack the flattened context vector to the flattened feature vector for the Conditioner in a conditional flow transform
            context_batch_shape = context.shape[:-len(self.context_shape)]
            assert x_batch_shape == context_batch_shape, f"Batch shape mismatch: features (x) has shape {x_batch_shape}, context has shape {context_batch_shape}"
            context = context.reshape(*context_batch_shape, -1)
            x = jnp.hstack([context,x])

        x = self._conditioner_mlp(x)
        x = self.activation(x)
        x = self._conditioner_out(x)
        x = x.reshape(*x_batch_shape, *(self.features_shape + (self.num_bijector_params,)))
        return x

class RQSplineFlow(nnx.Module):
    # Needed to make sure that non-nnx.Module objects from distrax are tracked by nnx
    # Alternative implementation is to use nnx.data(...) in the actual code and call on the 
    conditioners: nnx.List[Conditioner]
    transforms: nnx.Data[list]
    inverse_bijector: nnx.Data[ConditionalInverse]
    base_dist: nnx.Data[distrax.Distribution]
    inverse_flow: nnx.Data[ConditionalTransformed]

    def __init__(self,
                 n_features: int,
                 n_context: int = 0,
                 n_transforms: int = 4,
                 hidden_dims: tuple = ((32,32), (32,32)),
                 activation: str = "gelu",
                 n_bins: int = 8,
                 range_min: float = -1.0,
                 range_max: float = 1.0,
                 bijector_type: Callable = distrax.RationalQuadraticSpline,
                 ):
        
        self.features_shape = (n_features, )
        self.context_shape = (n_context, )
        self.n_bins = n_bins
        self.range_min = range_min
        self.range_max = range_max
        self.num_bijector_params = 3*self.n_bins + 1
        self.bijector_type = bijector_type
        self.n_transforms = n_transforms # Number of conditioner-bijector layers in the overall flow

        # Bijector transformation used in each layer of the flow
        # bijector_fn currently defined only for distrax.RationalQuadraticSpline
        def bijector_fn(params: Array):
            return self.bijector_type(params, self.range_min, self.range_max)
        self.bijector_fn = bijector_fn

        # Instantiate all the conditioners needed for #n_transform RQSpline transforms
        self.hidden_dims = hidden_dims
        self.conditioners = nnx.List() # Use nnx.List such that nnx knows to track the conditioner nnx modules
        for t in range(self.n_transforms):
            self.conditioners.append(
                Conditioner(self.features_shape, self.context_shape, self.hidden_dims, self.num_bijector_params)
            )

    # def make_flow(self):
        """
        Make distrax distribution containing the rational quadratic spline flow.

        Returns:
            Base Gaussian transformed by rational quadratic spline flow.
        """
        # First create alternating binary mask for the masked coupling mechanism 
        mask = jnp.arange(0, np.prod(self.features_shape)) % 2
        mask = jnp.reshape(mask, self.features_shape)
        self.mask = mask.astype(bool)

        # Now instantiate all the coupling trandforms
        # self.transforms = []
        # for t in range(self.n_transforms):
        #     self.transforms.append(
        #         ConditionalMaskedCoupling(mask=mask, bijector=self.bijector_fn, conditioner=self.conditioners[t])
        #         )
        #     mask = jnp.logical_not(mask) # Flipping the binary mask on the input features to the conditioner for each coupling transform
        # # self.transforms = nnx.data(self.transforms)

        # Now stack all transforms as single distrax.bjector child class 
        # Note that the convention used here is such that the RQSpline transforms map from data features to the conditional context,
        # We need to invert it for probability evaluation
        # inverse_flow and inverse_bijector refers to the flow transform mapping from: latent sampling distribution space --> feature space
        # self.inverse_bijector = ConditionalInverse(ConditionalChain(self.transforms))
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.features_shape), jnp.ones(self.features_shape))
        # self.inverse_flow = ConditionalTransformed(self.base_dist, self.inverse_bijector)

    def __call__(self, x: Array, context: Optional[Array] = None) -> Array:
        """
        Evaluate the log probability of the flow for non-batched input x.

        Args:
            x (jnp.ndarray (ndim)): Sample at which to predict posterior value.

        Returns:
            jnp.ndarray (float): Predicted log_e posterior value.
        """
        # self.make_flow()
        mask = self.mask
        transforms = []
        for t in range(self.n_transforms):
            transforms.append(
                ConditionalMaskedCoupling(mask=mask, bijector=self.bijector_fn, conditioner=self.conditioners[t])
                )
            mask = jnp.logical_not(mask) # Flipping the binary mask on the input features to the conditioner for each coupling transform
        inverse_bijector = ConditionalInverse(ConditionalChain(transforms))
        base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.features_shape), jnp.ones(self.features_shape))
        inverse_flow = ConditionalTransformed(self.base_dist, inverse_bijector)
        return inverse_flow.log_prob(x, context)

    def sample(self, num_samples: int, rng: Array, context: Array = None) -> Array:
        mask = self.mask
        transforms = []
        for t in range(self.n_transforms):
            transforms.append(
                ConditionalMaskedCoupling(mask=mask, bijector=self.bijector_fn, conditioner=self.conditioners[t])
                )
            mask = jnp.logical_not(mask) # Flipping the binary mask on the input features to the conditioner for each coupling transform
        inverse_bijector = ConditionalInverse(ConditionalChain(transforms))
        base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.features_shape), jnp.ones(self.features_shape))
        inverse_flow = ConditionalTransformed(self.base_dist, inverse_bijector)
        return inverse_flow.sample(seed=rng, sample_shape=(num_samples,), context=context)
