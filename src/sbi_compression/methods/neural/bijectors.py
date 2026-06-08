from typing import Any, List, Optional, Callable, Sequence, Tuple, Union
from jaxtyping import Array, Float, Int, PyTree # https://github.com/google/jaxtyping
from jax.random import PRNGKey

import jax
import jax.numpy as jnp

from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.bijectors.inverse import Inverse
from distrax._src.bijectors.chain import Chain
from distrax._src.distributions.transformed import Transformed


class ConditionalInverse(Inverse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.inverse(x, context)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.forward(y, context)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        return self._bijector.inverse_and_log_det(x, context)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        return self._bijector.forward_and_log_det(y, context)


class ConditionalMaskedCoupling(MaskedCoupling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        y, _ = self.forward_and_log_det(x, context=context)
        return y

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        x, _ = self.inverse_and_log_det(y, context=context)
        return x

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        self._check_forward_input_shape(x)
        masked_x = jnp.where(self._event_mask, x, 0.0)
        params = self._conditioner(masked_x, context)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = jnp.where(self._event_mask, x ,y0)
        # logdet = math.sum_last(
        #     jnp.where(self._mask, 0., log_d),
        #     self._event_ndims - self._inner_event_ndims
        # )
        # Or sum log-det Jacobian over event dimensions, robust to N-D feature(event) dimensions
        event_dims = tuple(range(-self._event_ndims, 0)) if self._event_ndims > 0 else ()
        logdet = jnp.sum(
            jnp.where(self._mask, 0., log_d),
            axis=event_dims
        )
        return y, logdet

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)
        params = self._conditioner(masked_y, context)
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        # logdet = math.sum_last(
        #     jnp.where(self._event_mask, 0., log_d),
        #     self._event_ndims - self._inner_event_ndims
        # )
        # Or sum log-det Jacobian over event dimensions, robust to N-D feature(event) dimensions
        event_dims = tuple(range(-self._event_ndims, 0)) if self._event_ndims > 0 else ()
        logdet = jnp.sum(
            jnp.where(self._mask, 0., log_d),
            axis=event_dims
        )
        return x, logdet


class ConditionalChain(Chain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, context)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        for bijector in self._bijectors:
            y = bijector.inverse(y, context)
        return y

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        x, log_det = self._bijectors[-1].forward_and_log_det(x, context)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        y, log_det = self._bijectors[0].inverse_and_log_det(y, context)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context)
            log_det += ld
        return y, log_det


class ConditionalTransformed(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def log_prob(self, y: Array, context: Optional[Array] = None) -> Array:
        """See `Distribution.log_prob`."""
        x, ildj_y = self.bijector.inverse_and_log_det(y, context=context)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample(self, seed: Union[int, PRNGKey], sample_shape: Tuple[int], context: Optional[Array] = None) -> Array:
        x = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        y, _ = self.bijector.forward_and_log_det(x, context)
        return y

    def sample_and_log_prob(self, seed: PRNGKey, sample_shape: List[int], context: Optional[Array] = None) -> Tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, context)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y