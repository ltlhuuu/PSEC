import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class StateValue(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init(), name="OutputVDense")(outputs)

        return jnp.squeeze(value, -1)
class StateValue_ws(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(2, kernel_init=default_init(), name="OutputVDense")(outputs)

        return value

class StateValue_ws3(nn.Module):
    base_cls: nn.Module

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(3, kernel_init=default_init(), name="OutputVDense")(outputs)

        return value