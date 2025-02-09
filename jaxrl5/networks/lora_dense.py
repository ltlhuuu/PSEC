import jax.numpy as jnp
from flax import linen as nn
# from flax.linen.module import Module, compact
# from flax.linen.dtypes import promote_dtype
import jax
from typing import Callable
import functools
from typing import (
  Any,
  Iterable,
  List,
  Optional,
  Sequence,
  Tuple,
  Union,
)

Initializer = Union[jax.nn.initializers.Initializer, Callable[..., Any]]

class LoRADense(nn.Module):
    features: int
    rank: int = 10
    lora_a_init: Callable = nn.initializers.lecun_normal()
    lora_b_init: Callable = nn.initializers.lecun_normal()
    lora_delta_init: Callable = nn.initializers.zeros


    @nn.compact
    def __call__(self, x, train_lora=False):
        # LoRA param
        lora_a = self.param('lora_a', self.lora_a_init, (x.shape[-1], self.rank))
        lora_b = self.param('lora_b', self.lora_b_init, (self.rank, self.features))
        x = jnp.dot(x, lora_a) @ lora_b
        return x
