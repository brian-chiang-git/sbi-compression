import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import flax
from flax import nnx 

import numpy as np
from math import prod
from typing import Callable, Sequence

class MLP(nnx.Module):
    def __init__(self,
                 layer_dims: Sequence[tuple],
                 activation: Callable = nnx.leaky_relu,
                 use_bias: bool = True,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 kernel_init: Callable = nnx.initializers.lecun_normal
                 ):
                 
        assert len(layer_dims)>0, "At least one layer dimension must be specified in layer_dims."
        self.activation = activation
        self.layer_dims = tuple(tuple(x) if not isinstance(x, tuple) else x for x in layer_dims)
        self.layers = nnx.List()
        for layer_dim in layer_dims:
            self.layers.append(nnx.Linear(layer_dim[0], 
                                          layer_dim[1], 
                                          rngs=rngs, 
                                          use_bias=use_bias, 
                                          kernel_init=kernel_init
                                          )
                                )
    def __call__(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

class NN_Build(nnx.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
        layer_dims: list,
        activation: Callable = nnx.leaky_relu,
        rngs: nnx.Rngs | None = None,
        ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.input_shape = tuple(input_shape)
        self.output_dim = output_dim
        self.layer_dims = tuple(tuple(x) if not isinstance(x, tuple) else x for x in layer_dims)
        if len(self.layer_dims) == 0:
            raise ValueError("At least one layer dimension must be specified in layer_dims.")
        self.activation = activation
        self.rngs = rngs
        self.layers = nnx.List()

        self.check_layer_dims(self.input_shape, self.output_dim, self.layer_dims)

        if len(self.layer_dims) == 1:
            self._add_layer(self.layer_dims[0], None, self.layers)#, final_layer=True)
        else:
            self._add_layer(self.layer_dims[0], None, self.layers)#, final_layer=False)
            size_prev = self.layer_dims[0]
            for size_new in self.layer_dims[1:-1]:
                self._add_layer(size_new, size_prev, self.layers)#, final_layer=False)
                size_prev = size_new
            self._add_layer(self.layer_dims[-1], size_prev, self.layers)#, final_layer=True)
            size_prev = self.layer_dims[-1]

    def check_layer_dims(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
        layer_dims: list,
        ):
        data_shape = tuple(input_shape)

        for layer_dim in layer_dims:
            if len(layer_dim) not in (2, 3):
                raise ValueError(
                    f"Each layer dimension must be a tuple of length 2 (linear) or 3 "
                    f"(convolutional). Got {layer_dim}."
                )

            if len(layer_dim) == 2:
                current_flat_dim = prod(data_shape)
                if current_flat_dim != layer_dim[0]:
                    raise ValueError(
                        f"Expected input shape {data_shape} to be compatible with linear "
                        f"layer input dimension {layer_dim[0]}."
                    )
                data_shape = (layer_dim[1],)

            elif len(layer_dim) == 3:
                if len(data_shape) != 3:
                    raise ValueError(
                        f"Cannot add a convolutional layer after a linear layer, got "
                        f"layer dimensions {layer_dim} after input shape {data_shape}."
                    )

                if data_shape[2] != layer_dim[0]:
                    raise ValueError(
                        f"Expected input shape {data_shape} to be compatible with "
                        f"convolutional layer input channels {layer_dim[0]}."
                    )

                new_h = data_shape[0] - layer_dim[2] + 1
                new_w = data_shape[1] - layer_dim[2] + 1
                if new_h <= 0 or new_w <= 0:
                    raise ValueError(
                        f"Convolution with kernel size {layer_dim[2]} is too large for "
                        f"spatial shape {data_shape[:2]}."
                    )

                data_shape = (new_h, new_w, layer_dim[1])

        if data_shape != (output_dim,):
            raise ValueError(
                f"Final layer shape {data_shape} does not match output dimension {output_dim}."
            )

    class Flatten(nnx.Module):
        def __call__(self, x):
            return x.reshape(x.shape[:-3] + (-1,))

    def _add_layer(
        self,
        layer_dim: tuple,
        prev_layer_dim: tuple | None,
        layers: nnx.List,
        # final_layer: bool = False,
        ):
        # class Flatten(nnx.Module):
        #     def __call__(self, x):
        #         return x.reshape((x.shape[0], -1))
        if prev_layer_dim is None:
            if len(layer_dim) == 2:
                if len(self.input_shape) == 1:
                    if self.input_shape[0] != layer_dim[0]:
                        raise ValueError(
                            f"Expected input shape {self.input_shape} to match linear "
                            f"input dimension {layer_dim[0]}."
                        )
                    layers.append(nnx.Linear(layer_dim[0], layer_dim[1], rngs=self.rngs))
                elif len(self.input_shape) == 3:
                    if prod(self.input_shape) != layer_dim[0]:
                        raise ValueError(
                            f"Expected flattened input size {prod(self.input_shape)} to "
                            f"match linear input dimension {layer_dim[0]}."
                        )
                    layers.append(self.Flatten())
                    layers.append(nnx.Linear(layer_dim[0], layer_dim[1], rngs=self.rngs))
                else:
                    raise ValueError(f"Unsupported input shape {self.input_shape}.")

            elif len(layer_dim) == 3:
                if len(self.input_shape) != 3:
                    raise ValueError(
                        f"Cannot add a convolutional layer after a linear layer, got "
                        f"layer dimensions {layer_dim} after input shape {self.input_shape}."
                    )
                if self.input_shape[2] != layer_dim[0]:
                    raise ValueError(
                        f"Expected input channels {self.input_shape[2]} to match "
                        f"convolutional input channels {layer_dim[0]}."
                    )
                layers.append(
                    nnx.Conv(
                        layer_dim[0],
                        layer_dim[1],
                        kernel_size=(layer_dim[2], layer_dim[2]),
                        padding="VALID",
                        rngs=self.rngs,
                    )
                )   

        else:
            if len(layer_dim) == 2 and len(prev_layer_dim) == 2:
                if prev_layer_dim[1] != layer_dim[0]:
                    raise ValueError(
                        f"Linear layer input dimension {layer_dim[0]} does not match "
                        f"previous layer output dimension {prev_layer_dim[1]}."
                    )
                layers.append(nnx.Linear(layer_dim[0], layer_dim[1], rngs=self.rngs))

            elif len(layer_dim) == 3 and len(prev_layer_dim) == 3:
                if prev_layer_dim[1] != layer_dim[0]:
                    raise ValueError(
                        f"Convolutional layer input channels {layer_dim[0]} do not match "
                        f"previous layer output channels {prev_layer_dim[1]}."
                    )
                layers.append(
                    nnx.Conv(
                        layer_dim[0],
                        layer_dim[1],
                        kernel_size=(layer_dim[2], layer_dim[2]),
                        padding="VALID",
                        rngs=self.rngs,
                    )
                )

            elif len(layer_dim) == 2 and len(prev_layer_dim) == 3:
                layers.append(self.Flatten())
                layers.append(nnx.Linear(layer_dim[0], layer_dim[1], rngs=self.rngs))

            elif len(layer_dim) == 3 and len(prev_layer_dim) == 2:
                raise ValueError(
                    f"Cannot add a convolutional layer after a linear layer, got "
                    f"layer dimensions {layer_dim} after {prev_layer_dim}."
                )

        # if not final_layer:
        #     layers.append(self.activation)

    def __call__(self, x):
        if len(self.layers) > 1:
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.activation(x)
            x = self.layers[-1](x)
        elif len(self.layers) == 1:
            for layer in self.layers:
                x = layer(x)
        return x