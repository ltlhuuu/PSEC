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
# from flax.linen import initializers
Initializer = Union[jax.nn.initializers.Initializer, Callable[..., Any]]
class LoRALayer(nn.Module):
    features: int
    rank: int  
    alpha_r: int 
    pretrain_kernel: Optional[jnp.ndarray] = None  # 预训练的 kernel
    pretrain_bias: Optional[jnp.ndarray] = None    # 预训练的 bias
    lora_a0: Optional[jnp.ndarray] = None
    lora_b0: Optional[jnp.ndarray] = None
    
    lora_a_init: Callable = nn.initializers.lecun_normal()
    lora_b_init: Callable = nn.initializers.zeros

    
    @nn.compact
    def __call__(self, x, alpha_as):
        
        lora_a = self.param('lora_a', self.lora_a_init, (x.shape[-1], self.rank))
        lora_b = self.param('lora_b', self.lora_b_init, (self.rank, self.features))
        
        out = jnp.dot(x, self.pretrain_kernel) + self.pretrain_bias
        delta_w0 = jnp.dot(x, self.lora_a0) @ self.lora_b0
        delta_w = jnp.dot(x, lora_a) @ lora_b

        if alpha_as is not None:
            out += alpha_as[:, 0][:, None] * delta_w0 + alpha_as[:, 1][:, None] * delta_w
        else:
            out += self.alpha_r * delta_w0 + self.alpha_r * delta_w
        # out += self.alpha_r * delta_w0
        # out += self.alpha_r * delta_w

        return out
    

class LoRA_MLP(nn.Module):
    pretrain_weights: dict 
    lora0: dict
    rank: int
    alpha_r: int
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, alpha_as: jnp.ndarray = None, training: bool = False) -> jnp.ndarray:
        if self.use_layer_norm:
            # x = nn.LayerNorm()(x)
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.var(x, axis=-1, keepdims=True)
            
            x_normalized = (x - mean) / jnp.sqrt(variance + 1e-6)
            scale=self.pretrain_weights['target_score_model']['params']['MLP_0']['LayerNorm_0']['scale']
            bias=self.pretrain_weights['target_score_model']['params']['MLP_0']['LayerNorm_0']['bias']
            x = scale * x_normalized + bias
            
        for i, size in enumerate(self.hidden_dims):
            # if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
            x = LoRALayer(size, rank=self.rank, alpha_r=self.alpha_r,
                            pretrain_kernel=self.pretrain_weights['target_score_model']['params']['MLP_0'][f'Dense_{i}']['kernel'],
                            pretrain_bias=self.pretrain_weights['target_score_model']['params']['MLP_0'][f'Dense_{i}']['bias'],
                            lora_a0=self.lora0['q_target_lora_model']['params']['LoRA_MLP_0'][f'LoRALayer_{i}']['lora_a'],
                            lora_b0=self.lora0['q_target_lora_model']['params']['LoRA_MLP_0'][f'LoRALayer_{i}']['lora_b'],)(x, alpha_as=alpha_as)


            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                x = self.activations(x)
        return x


class LoRAResNet(nn.Module):
    pretrain_weights: dict 
    lora0: dict
    rank: int
    alpha_r: int
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu


    @nn.compact
    def __call__(self, x: jnp.ndarray, alpha_as: jnp.ndarray = None, training: bool = False) -> jnp.ndarray:

        # lora dense
        x = LoRALayer(self.hidden_dim, rank=self.rank, alpha_r=self.alpha_r,
                      pretrain_kernel=self.pretrain_weights['target_score_model']['params']['MLPResNet_0']['Dense_0']['kernel'],
                      pretrain_bias=self.pretrain_weights['target_score_model']['params']['MLPResNet_0']['Dense_0']['bias'],
                      lora_a0=self.lora0['q_target_lora_model']['params']['LoRAResNet_0']['LoRALayer_0']['lora_a'],
                      lora_b0=self.lora0['q_target_lora_model']['params']['LoRAResNet_0']['LoRALayer_0']['lora_b'],
                      )(x, alpha_as=alpha_as)
        # add lora run
        for i in range(self.num_blocks):
            x = LoRAResnetBlock(self.hidden_dim, act=self.activations, use_layer_norm=self.use_layer_norm, dropout_rate=self.dropout_rate)(x, i, rank=self.rank, alpha_r=self.alpha_r, alpha_as=alpha_as, pretrain_weights=self.pretrain_weights, lora0=self.lora0, training=training)
        x = self.activations(x)
        # lora dense
        x = LoRALayer(self.out_dim, rank=self.rank, alpha_r=self.alpha_r,
                      pretrain_kernel=self.pretrain_weights['target_score_model']['params']['MLPResNet_0']['Dense_1']['kernel'],
                      pretrain_bias=self.pretrain_weights['target_score_model']['params']['MLPResNet_0']['Dense_1']['bias'],
                      lora_a0=self.lora0['q_target_lora_model']['params']['LoRAResNet_0']['LoRALayer_1']['lora_a'],
                      lora_b0=self.lora0['q_target_lora_model']['params']['LoRAResNet_0']['LoRALayer_1']['lora_b'],)(x, alpha_as=alpha_as)

        return x
    

class LoRAResnetBlock(nn.Module):
    """LoRAResnetBlock"""
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False
    

    @nn.compact
    def __call__(self, x, i,  rank, alpha_r,  pretrain_weights, lora0, alpha_as: jnp.ndarray = None, training: bool = False,):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(
                x, deterministic=not training
            )
        if self.use_layer_norm:
            # norm = nn.LayerNorm()(x)
            # 计算均值和方差
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.var(x, axis=-1, keepdims=True)
            
            # 标准化
            x_normalized = (x - mean) / jnp.sqrt(variance + 1e-6)
            scale=pretrain_weights['target_score_model']['params']['MLPResNet_0'][f'MLPResNetBlock_{i}']['LayerNorm_0']['scale']
            bias=pretrain_weights['target_score_model']['params']['MLPResNet_0'][f'MLPResNetBlock_{i}']['LayerNorm_0']['bias']
            x = scale * x_normalized + bias

            # x = nn.LayerNorm()(x)

        x = LoRALayer(self.features * 4, rank=rank, alpha_r=alpha_r,
                      pretrain_kernel=pretrain_weights['target_score_model']['params']['MLPResNet_0'][f'MLPResNetBlock_{i}']['Dense_0']['kernel'],
                      pretrain_bias=pretrain_weights['target_score_model']['params']['MLPResNet_0'][f'MLPResNetBlock_{i}']['Dense_0']['bias'],
                      lora_a0=lora0['q_target_lora_model']['params']['LoRAResNet_0'][f'LoRAResnetBlock_{i}']['LoRALayer_0']['lora_a'],
                      lora_b0=lora0['q_target_lora_model']['params']['LoRAResNet_0'][f'LoRAResnetBlock_{i}']['LoRALayer_0']['lora_b'],)(x, alpha_as=alpha_as)
        
        x = self.act(x)

        x = LoRALayer(self.features, rank=rank, alpha_r=alpha_r,
                      pretrain_kernel=pretrain_weights['target_score_model']['params']['MLPResNet_0'][f'MLPResNetBlock_{i}']['Dense_1']['kernel'],
                      pretrain_bias=pretrain_weights['target_score_model']['params']['MLPResNet_0'][f'MLPResNetBlock_{i}']['Dense_1']['bias'],
                      lora_a0=lora0['q_target_lora_model']['params']['LoRAResNet_0'][f'LoRAResnetBlock_{i}']['LoRALayer_1']['lora_a'],
                      lora_b0=lora0['q_target_lora_model']['params']['LoRAResNet_0'][f'LoRAResnetBlock_{i}']['LoRALayer_1']['lora_b'],)(x, alpha_as=alpha_as)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)
            
        return residual + x

class LoRA_FourierFeatures(nn.Module):
    pretrain_weights: dict 
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            # w = self.param('kernel', nn.initializers.normal(0.2),
            #                (self.output_size // 2, x.shape[-1]), jnp.float32)
            w = self.pretrain_weights['target_score_model']['params']['FourierFeatures_0']['kernel']
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)