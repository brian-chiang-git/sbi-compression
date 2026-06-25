
from typing import Literal, Callable, Optional
from jaxtyping import Array

import flax
from flax import nnx
import jax.numpy as jnp
import distrax
from math import prod

from .flows import RQSplineFlow
from .nn_nnx import NN_Build

class AutoencoderFlow(nnx.Module):

    def __init__(self,
                 input_shape: tuple,
                 features_shape: tuple,
                 context_shape: tuple,
                 encoder_layer_dims: tuple,
                 conditioner_hidden_dims: tuple,
                 encoder_mode: Literal['context', 'features'],
                 activation: str = 'gelu',
                 n_transforms: int = 4,
                 n_bins: int = 8,
                 range_min: float = -10.0,
                 range_max: float = 10.0,
                 bijector_type: Callable = distrax.RationalQuadraticSpline,
                 ):

        self.activation = activation
        self.encoder_mode = encoder_mode

        self.input_shape = input_shape
        self.features_shape = features_shape
        self.features_dim = prod(features_shape)
        self.context_shape = context_shape
        self.context_dim = prod(context_shape)
        self.encoder_layer_dims = encoder_layer_dims
        self.encoder = NN_Build(self.input_shape, 
                                self.features_dim, 
                                self.encoder_layer_dims, 
                                activation=getattr(nnx, self.activation))

        self.n_transforms = n_transforms
        self.n_bins = n_bins
        self.range_min = range_min
        self.range_max = range_max
        self.bijector_type = bijector_type
        self.conditioner_hidden_dims = conditioner_hidden_dims
        self.flow = RQSplineFlow(self.features_dim, 
                                 self.context_dim, 
                                 n_transforms=self.n_transforms, 
                                 hidden_dims=self.conditioner_hidden_dims, 
                                 activation=getattr(nnx, self.activation), 
                                 n_bins=self.n_bins, 
                                 range_min=self.range_min, 
                                 range_max=self.range_max, 
                                 bijector_type=self.bijector_type)
    
    def __call__(self, x: Array, context: Array) -> Array:
        if self.encoder_mode == 'context':
            context_encoded = self.encoder(context)
            x = self.flow(x, context_encoded)
        elif self.encoder_mode == 'features':
            x_encoded = self.encoder(x)
            x = self.flow(x_encoded, context)
        return x
    
    def sample(self, num_samples: int, rng: Array, context: Array) -> Array:
        x = self.flow.sample(num_samples, rng, context)
        return x
        
    def encode(self, x: Array) -> Array:
        assert x.shape[-len(self.input_shape):] == self.input_shape # Make sure the input shape matches the encoder, assume batch first
        x = self.encoder(x)
        return x
    
    def mode(self, mode: Literal['train_encoder', 'train_flow', 'train_all', 'eval']) -> None:
        if mode == 'train_encoder':
            self.encoder.train()
            self.flow.eval()
        elif mode == 'train_flow':
            self.encoder.eval()
            self.flow.train()
        elif mode == 'train_all':
            self.encoder.train()
            self.flow.train()
        elif mode == 'eval':
            self.encoder.eval()
            self.flow.eval()
