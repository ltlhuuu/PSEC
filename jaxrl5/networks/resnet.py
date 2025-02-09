from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import flax
from jaxrl5.networks.lora_dense import LoRADense
import jax
default_init = nn.initializers.xavier_uniform

class MLPResNetBlock_lora(nn.Module):
    """MLPResNet block."""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False
    use_lora: bool = True
    @nn.compact
    def __call__(self, x, training: bool = False, train_lora: bool = True):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x) if not self.use_lora else LoRADense(self.features * 4)(x, train_lora=train_lora)
        x = self.act(x)
        x = nn.Dense(self.features)(x) if not self.use_lora else LoRADense(self.features)(x, train_lora=train_lora)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


# # pretrain weight block
# class MLPResNetBlock(nn.Module):
#     """MLPResNet block."""
#     features: int
#     act: Callable
#     dropout_rate: float = None
#     use_layer_norm: bool = False
#     use_lora: bool = True
#     @nn.compact
#     def __call__(self, x, training: bool = False, train_lora: bool = True):
#         residual = x
#         if self.dropout_rate is not None and self.dropout_rate > 0.0:
#             x = nn.Dropout(rate=self.dropout_rate)(
#                 x, deterministic=not training)
#         if self.use_layer_norm:
#             x = nn.LayerNorm()(x)
#         x = nn.Dense(self.features * 4)(x)
#         x = self.act(x)
#         x = nn.Dense(self.features)(x)

#         if residual.shape != x.shape:
#             residual = nn.Dense(self.features)(residual)

#         return residual + x
    
# # add lora AB block
class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)
        # if self.use_lora:
        #     x += LoRADense(self.features * 4)(x, train_lora=train_lora)
        # if self.use_lora:
        #     x += LoRADense(self.features)(x, train_lora=train_lora)
        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x

class LoRAResnetBlock(nn.Module):
    """LoRAResnetBlock"""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training
            )
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = LoRADense(self.features * 4)(x)
        x = self.act(x)
        x = LoRADense(self.features)(x)
        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)
        return residual + x
    
class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False, stop_gradient: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, training=training)
        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if stop_gradient:
            x = jax.lax.stop_gradient(x)
        return x
class MLPResNet1(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu
    use_lora: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False, train_lora: bool = False, stop_gradient: bool = False, weight: float = 0.1) -> jnp.ndarray:
        # x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x) if not self.use_lora else LoRADense(self.hidden_dim, kernel_init=default_init())(x, train_lora=train_lora)
        
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate, use_lora=self.use_lora)(x, training=training, train_lora=train_lora)
        x = self.activations(x)
        # x = nn.Dense(self.out_dim, kernel_init=default_init())(x) if not self.use_lora else LoRADense(self.out_dim, kernel_init=default_init())(x, train_lora=train_lora)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if stop_gradient:
            x = jax.lax.stop_gradient(x)
        return x

class MLPResNet2(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu
    use_lora: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False, train_lora: bool = False, stop_gradient: bool = False, weight: float = 0.1) -> jnp.ndarray:
        # x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x) if not self.use_lora else LoRADense(self.hidden_dim, kernel_init=default_init())(x, train_lora=train_lora)
        
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate, use_lora=self.use_lora)(x, training=training, train_lora=train_lora)
        x = self.activations(x)
        # x = nn.Dense(self.out_dim, kernel_init=default_init())(x) if not self.use_lora else LoRADense(self.out_dim, kernel_init=default_init())(x, train_lora=train_lora)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if stop_gradient:
            x = jax.lax.stop_gradient(x)
        return x


class LoRAResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # lora dense
        x = LoRADense(self.hidden_dim)(x)
        
        for _ in range(self.num_blocks):
            x = LoRAResnetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, training=training)
        x = self.activations(x)
        # lora dense
        x = LoRADense(self.out_dim)(x)
        return x

# combine the LoRAResNet and the MLPResNet
class combineResnet(nn.Module):
    
    @nn.compact
    def __call__(self, pretrain_x, lora_x=0.0):
        x = pretrain_x + lora_x
        return x
    
        

    

    