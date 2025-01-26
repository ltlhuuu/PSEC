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

class LoRADense(nn.Module):
    features: int
    rank: int = 20
    # kernel_init = default_kernel_init
    # bias_init = initializers.zeros_init()
    lora_a_init: Callable = nn.initializers.lecun_normal()
    lora_b_init: Callable = nn.initializers.lecun_normal()
    lora_delta_init: Callable = nn.initializers.zeros
    # add the init of the lora_delta
    # def setup(self):
    #     lora_delta = self.param('lora_delta', self.lora_delta_init, (self.features,))

    @nn.compact
    def __call__(self, x):
        # LoRA param
        # lora_delta = self.param('lora_delta', self.lora_delta_init, (self.features,))
        lora_a = self.param('lora_a', self.lora_a_init, (x.shape[-1], self.rank))
        lora_b = self.param('lora_b', self.lora_b_init, (self.rank, self.features))
        # x, lora_a, lora_b = promote_dtype(x, lora_a, lora_b)
    #     bias = self.param(
    #     'bias', self.bias_init, (self.features,), self.param_dtype
    #   )
        # lora_a = self.param('lora_a', self.kernel_init, (x.shape[-1], self.rank))
        # lora_b = self.param('lora_b', self.bias_init, (self.rank, self.features))
        x = jnp.dot(x, lora_a) @ lora_b
        return x