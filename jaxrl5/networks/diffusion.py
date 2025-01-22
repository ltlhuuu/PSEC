from functools import partial
from typing import Callable, Optional, Sequence, Type
import flax.linen as nn
import jax.numpy as jnp
import jax

beta_1 = 20.0
beta_0 = 0.1
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(
        beta_start, beta_end, timesteps
    )
    return betas

def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas
def vp_sde_schedule(t):
    """Continous VPSDE schedule. Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = jnp.exp(log_mean_coeff)
    std = jnp.sqrt(1. - jnp.exp(2. * log_mean_coeff))
    return alpha_t, std

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
class FourierFeatures1(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
class FourierFeatures2(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
class lora_FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

# TODO: reduce the s
class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False,):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, training=training)
    
class DDPM_alpha(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 alpha_as: jnp.ndarray = None,
                 training: bool = False):
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, alpha_as=alpha_as, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, alpha_as=alpha_as, training=training)
    
class DDPM_demo(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False,
                 train_lora: bool = False,
                 stop_gradient: bool = False,
                 weight: float = 0.1):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, training=training, train_lora=train_lora, stop_gradient=stop_gradient, weight=weight)


def dpm_solver_first_update(x, s, t, eps_pred):
    # dims = x.dim()
    lambda_s, lambda_t = marginal_lambda(s), marginal_lambda(t)
    h = lambda_t - lambda_s
    log_alpha_s, log_alpha_t = marginal_log_mean_coeff(s), marginal_log_mean_coeff(t)
    sigma_t = marginal_std(t)

    phi_1 = jnp.expm1(h)

    x_t = (
            (jnp.exp(log_alpha_t - log_alpha_s)) * x
            - (sigma_t * phi_1) * eps_pred
    )
    return x_t
@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def dpm_solver_sampler_1st(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                           sample_temperature, repeat_last_step, clip_sampler, training=False):
    batch_size = observations.shape[0]
    t_T = 1.
    t_0 = 1e-3
    time_steps = jnp.linspace(t_T, t_0, T + 1)
    orders = [1, ] * T  # first order solver

    def singlestep_dpm_solver_update(input_tuple, time_index):
        current_x, rng = input_tuple
        vec_s = jnp.expand_dims(jnp.array([time_steps[time_index]]).repeat(current_x.shape[0]), axis=1)
        vec_t = jnp.expand_dims(jnp.array([time_steps[time_index + 1]]).repeat(current_x.shape[0]), axis=1)

        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, vec_s, training=training)

        current_x = dpm_solver_first_update(current_x, vec_s, vec_t, eps_pred)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        rng, key = jax.random.split(rng, 2)
        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(singlestep_dpm_solver_update,
                                   (jax.random.normal(key, (batch_size, act_dim)) * 0, rng), jnp.arange(0, T, 1),
                                   unroll=5)

    for _ in range(repeat_last_step):
        input_tuple, () = singlestep_dpm_solver_update(input_tuple, T)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)
    return action_0, rng

@partial(jax.jit, static_argnames=('actor_apply_fn', 'clip_sampler', 'training', 'train_lora'))
def fn_test(input_tuple, actor_apply_fn, actor_params, rng, observations, alphas, alpha_hats, betas, sample_temperature, clip_sampler, rngs, training, train_lora, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time,
                                  rngs=rngs,
                                  training=training,
                                  train_lora=train_lora)
        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key1 = jax.random.split(rng, 2)
        z = jax.random.normal(key1,
                            shape=(observations.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        current_x = jax.lax.stop_gradient(current_x)

        return (current_x, rng), (), time

def ddpm_sampler_test(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, train_lora=False):

    batch_size = observations.shape[0]

    key, rng = jax.random.split(rng, 2)

    input_tuple, (), time = jax.lax.scan(fn_test, (jax.random.normal(key, (batch_size, act_dim)), rng),actor_apply_fn, actor_params, rng, observations, alphas, alpha_hats, betas, sample_temperature, clip_sampler, rngs, training, train_lora, jnp.arange(T-1, 0, -1))

    for _ in range(repeat_last_step):
        input_tuple, () = fn_test(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_multisample(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, train_lora=False, N=8):
    
    observations = jnp.expand_dims(observations, axis=0).repeat(8, axis = 0)
    observations = observations.reshape(-1, observations.shape[-1])
    batch_size = observations.shape[0]
    key, rng = jax.random.split(rng, 2)
    def fn1(input_tuple, time):

        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time,
                                  rngs=rngs,
                                  training=training,
                                  train_lora=False)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key1 = jax.random.split(rng, 2)
        z = jax.random.normal(key1,
                              shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()
    def fn2(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time,
                                  rngs=rngs,
                                  training=training,
                                  train_lora=True)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key1 = jax.random.split(rng, 2)
        z = jax.random.normal(key1, shape=(observations.shape[0], current_x.shape[1]), )
        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    input_tuple, () = jax.lax.scan(fn1, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, 0, -1))

    for _ in range(repeat_last_step):
        input_tuple, () = fn2(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng
@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn'))
def ddpm_sampler_first(q1_actor_apply_fn, q1_actor_params, score_apply_fn, score_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, weight=0.1):

    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, observations, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, observations, current_x, input_time, 
                                             rngs=rngs, 
                                             training=True, 
                                             train_lora=True,
                                             stop_gradient=False,
                                             weight=weight)

        eps_pred = jax.lax.cond(time == 0,
                        lambda _: eps_pred + q1_eps_pred_lora,
                        lambda _: eps_pred,
                        operand=None)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(observations.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)


        return (current_x, rng), ()
    
    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn', 'q2_actor_apply_fn'))
def ddpm_sampler(q1_actor_apply_fn, q1_actor_params, q2_actor_apply_fn, q2_actor_params, score_apply_fn, score_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, weight=0.1):

    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, observations, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, observations, current_x, input_time, 
                                             rngs=rngs, 
                                             training=True, 
                                             train_lora=True,
                                             stop_gradient=False,
                                             weight=weight)
        q2_eps_pred_lora = q2_actor_apply_fn({"params": q2_actor_params}, observations, current_x, input_time, 
                                             training=False, 
                                             train_lora=True,
                                             stop_gradient=True)

        eps_pred = jax.lax.cond(time == 1,
                        lambda _: eps_pred + q2_eps_pred_lora,
                        lambda _: eps_pred,
                        operand=None)
        
        eps_pred = jax.lax.cond(time == 0,
                        lambda _: eps_pred + q1_eps_pred_lora,
                        lambda _: eps_pred,
                        operand=None)
        
        # eps_pred = eps_pred_lora + eps_pred_pretrain

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(observations.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)


        return (current_x, rng), ()
    
    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, -1, -1))


    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

@partial(jax.jit, static_argnames=( 'reward_actor_apply_fn', 'act_dim', 'cost_actor_apply_fn', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_ws(ws, 
                        reward_actor_apply_fn, reward_actor_params, 
                        cost_actor_apply_fn, cost_actor_params, 
                        T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        reward_eps_pred = reward_actor_apply_fn({"params": reward_actor_params}, observations, current_x, input_time,
                                  training=training,)
        cost_eps_pred = cost_actor_apply_fn({"params": cost_actor_params}, observations, current_x, input_time,
                                          training=training,)
        eps_pred = ws[:, 0][:, None] * reward_eps_pred + ws[:, 1][:, None] * cost_eps_pred

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1.0, 1.0)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.0, 1.0)

    return action_0, rng

@partial(jax.jit, static_argnames=('BC_actor_apply_fn', 'reward_actor_apply_fn', 'act_dim', 'cost_actor_apply_fn', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_ws3(ws, BC_actor_apply_fn, BC_actor_params,
                        reward_actor_apply_fn, reward_actor_params, 
                        cost_actor_apply_fn, cost_actor_params, 
                        T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        BC_eps_pred = BC_actor_apply_fn({"params": BC_actor_params}, observations, current_x, input_time, training=training)
        reward_eps_pred = reward_actor_apply_fn({"params": reward_actor_params}, observations, current_x, input_time,
                                  training=training,)
        cost_eps_pred = cost_actor_apply_fn({"params": cost_actor_params}, observations, current_x, input_time,
                                          training=training,)
        eps_pred = ws[:, 0][:, None] * BC_eps_pred + ws[:, 1][:, None] * reward_eps_pred + ws[:, 2][:, None] * cost_eps_pred

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1.0, 1.0)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.0, 1.0)

    return action_0, rng

@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn', 'q2_actor_apply_fn'))
def ddpm_sampler_eval(q1_actor_apply_fn, q1_actor_params, q2_actor_apply_fn, q2_actor_params, score_apply_fn, score_params, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False, train_lora=False, train_maxq_weight=0.07, train_minqc_weight=0.07, eval_maxq_weight=0.07, eval_minqc_weight=0.07):
    batch_size = observations.shape[0]
    
    def fn(input_tuple, time):

        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, observations, current_x, input_time, training=False, train_lora=False, )
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, observations, current_x, input_time, training=training, train_lora=True, weight=train_maxq_weight)
        q2_eps_pred_lora = q2_actor_apply_fn({"params": q2_actor_params}, observations, current_x, input_time, training=training, train_lora=True, weight=train_minqc_weight)
        
        # compute the eps_pred_lora l2 distance
        q1_eps_pred_lora_dis= jnp.linalg.norm(q1_eps_pred_lora, axis=-1, keepdims=True).sum().mean()
        q2_eps_pred_lora_dis= jnp.linalg.norm(q2_eps_pred_lora, axis=-1, keepdims=True).sum().mean()
        # compute the eps_pred l2 distance
        eps_pred_dis = jnp.linalg.norm(eps_pred, axis=-1, keepdims=True).sum().mean()

        # scale_factor = eps_pred_dis.mean() / eps_pred_lora_dis.mean()
        # eps_pred_lora_scaled = eps_pred_lora * scale_factor
        # eps_pred =  eps_pred + eps_pred_lora

        eps_pred = jax.lax.cond(time == 0,
                        lambda _: eps_pred + eval_maxq_weight * q1_eps_pred_lora + eval_minqc_weight * q2_eps_pred_lora,
                        lambda _: eps_pred,
                        operand=None)

        # eps_pred_pretrain = score_apply_fn({"params": score_params}, observations, current_x, input_time,
        #                             training=training,
        #                             train_lora=False)
        # eps_pred = eps_pred_lora + eps_pred_pretrain

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), (q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis)
    
    rng, key = jax.random.split(rng, 2)
    input_tuple, outputs = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))
    q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = outputs
    # for _ in range(repeat_last_step):
    #     input_tuple, () = fn(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_eval1(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False, train_lora=False):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time,
                                  training=training, train_lora=True)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_eval_bc(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time,
                                  training=training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng
@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_eval_alphas(actor_apply_fn, actor_params, alpha_as, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, alpha_as=alpha_as,
                                  training=training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(observations.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1.0, 1.0)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.0, 1.0)

    return action_0, rng


@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_ori(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False):
    batch_size = observations.shape[0]

    def fn(input_tuple, time):
        # only train lora in the specific layer
        train_lora = jax.lax.cond(time >= 1, lambda _: False, lambda _: True, operand=None)

        current_x, rng = input_tuple
        key, rng = jax.random.split(rng, 2)
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, rngs={'dropout': key},training=training, train_lora=train_lora)
        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        # rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(observations.shape[0], current_x.shape[1]), )

        # cut down the gradients in time = 1,2,3,4
        z = jax.lax.cond(time >= 1, lambda _: jax.lax.stop_gradient(z), lambda _: z, operand=None)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    for _ in range(repeat_last_step):
        input_tuple, () = fn(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng

def marginal_lambda(t):
    """
    Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
    """
    log_mean_coeff = marginal_log_mean_coeff(t)
    log_std = 0.5 * jnp.log(1. - jnp.exp(2. * log_mean_coeff))
    return log_mean_coeff - log_std

def marginal_log_mean_coeff(t):
    return -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0

def marginal_std(t):
    """
    Compute sigma_t of a given continuous-time label t in [0, T].
    """
    return jnp.sqrt(1. - jnp.exp(2. * marginal_log_mean_coeff(t)))

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a jnp.DeviceArray with shape [N].
        `dim`: a `int`.
    Returns:
        a jnp.DeviceArray with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]

@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn',))
def ddpm_sampler_first_demo(q1_actor_apply_fn, q1_actor_params, score_apply_fn, score_params, T, rng, act_dim, actions, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, weight=0.1):

    batch_size = actions.shape[0]

    def fn1(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        # current_x = jax.lax.stop_gradient(current_x)

        return (current_x, rng), ()
    
    def fn2(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, current_x, input_time, 
                                             rngs=rngs, 
                                             training=True, 
                                             train_lora=True,
                                             stop_gradient=False,
                                             weight=weight)

        eps_pred = eps_pred + q1_eps_pred_lora
                        

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        # current_x = jax.lax.stop_gradient(current_x)

        return (current_x, rng), ()
    
    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn2, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, -1, -1))
    # input_tuple, () = jax.lax.scan(fn2, input_tuple, jnp.arange(20, -1, -1))
    # input_tuple, () = fn2(input_tuple, 0)

    # for _ in range(repeat_last_step):
    #     input_tuple, () = fn(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.5, 1.5)

    return action_0, rng

@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn',))
def ddpm_sampler_minus_demo(q1_actor_apply_fn, q1_actor_params, score_apply_fn, score_params, T, rng, act_dim, actions, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, weight=0.1):

    batch_size = actions.shape[0]

    def fn1(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        # current_x = jax.lax.stop_gradient(current_x)

        return (current_x, rng), ()
    
    def fn2(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, current_x, input_time, 
                                             rngs=rngs, 
                                             training=True, 
                                             train_lora=True,
                                             stop_gradient=False,
                                             weight=weight)

        eps_pred = eps_pred - q1_eps_pred_lora
                        

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        # current_x = jax.lax.stop_gradient(current_x)

        return (current_x, rng), ()
    
    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn1, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, 20, -1))
    input_tuple, () = jax.lax.scan(fn2, input_tuple, jnp.arange(20, -1, -1))
    # input_tuple, () = fn2(input_tuple, 0)

    # for _ in range(repeat_last_step):
    #     input_tuple, () = fn(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.5, 1.5)

    return action_0, rng

# lora single eps
@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn',))
def ddpm_sampler_first_demo_single(q1_actor_apply_fn, q1_actor_params, score_apply_fn, score_params, T, rng, act_dim, actions, alphas, alpha_hats, betas, sample_temperature, repeat_last_step, clip_sampler, rngs, training=False, weight=0.1):

    batch_size = actions.shape[0]

    def fn1(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        # current_x = jax.lax.stop_gradient(current_x)

        return (current_x, rng), ()
    
    def fn2(input_tuple, time):
        current_x, current_x_lora, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time,
                                training=False,
                                train_lora=False, 
                                stop_gradient=True)
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, current_x_lora, input_time, 
                                             rngs=rngs, 
                                             training=True, 
                                             train_lora=True,
                                             stop_gradient=False,
                                             weight=weight)
                        

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)
        current_x_lora = alpha_1 * (current_x - alpha_2 * q1_eps_pred_lora)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]),)

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)
        current_x_lora = current_x_lora + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        # cut the gradient
        # current_x = jax.lax.stop_gradient(current_x)

        return (current_x, current_x_lora, rng), ()
    
    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn2, (jax.random.normal(key, (batch_size, act_dim)), jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T-1, -1, -1))
    # input_tuple, () = jax.lax.scan(fn2, input_tuple, jnp.arange(20, -1, -1))
    # input_tuple, () = fn2(input_tuple, 0)

    # for _ in range(repeat_last_step):
    #     input_tuple, () = fn(input_tuple, 0)

    action_0, action_0_lora, rng = input_tuple
    action_0 = action_0 + action_0_lora
    action_0 = jnp.clip(action_0, -1.5, 1.5)

    return action_0, rng


@partial(jax.jit, static_argnames=('q1_actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training', 'score_apply_fn', 'q2_actor_apply_fn'))
def ddpm_sampler_eval_demo(q1_actor_apply_fn, q1_actor_params, q2_actor_apply_fn, q2_actor_params, score_apply_fn, score_params, T, rng, act_dim, actions, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False, train_lora=False, train_maxq_weight=0.07, train_minqc_weight=0.07, eval_maxq_weight=0.07, eval_minqc_weight=0.07):
    batch_size = actions.shape[0]
    
    def fn(input_tuple, time):

        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)

        eps_pred = score_apply_fn({"params": score_params}, current_x, input_time, training=False, train_lora=False, )
        
        q1_eps_pred_lora = q1_actor_apply_fn({"params": q1_actor_params}, current_x, input_time, training=training, train_lora=True, weight=train_maxq_weight)
        q2_eps_pred_lora = q2_actor_apply_fn({"params": q2_actor_params}, current_x, input_time, training=training, train_lora=True, weight=train_minqc_weight)
        
        # compute the eps_pred_lora l2 distance
        q1_eps_pred_lora_dis= jnp.linalg.norm(q1_eps_pred_lora, axis=-1, keepdims=True).sum().mean()
        q2_eps_pred_lora_dis= jnp.linalg.norm(q2_eps_pred_lora, axis=-1, keepdims=True).sum().mean()
        # compute the eps_pred l2 distance
        eps_pred_dis = jnp.linalg.norm(eps_pred, axis=-1, keepdims=True).sum().mean()

        # scale_factor = eps_pred_dis.mean() / eps_pred_lora_dis.mean()
        # eps_pred_lora_scaled = eps_pred_lora * scale_factor
        # eps_pred =  eps_pred + eps_pred_lora

        # eps_pred = jax.lax.cond(time == 0,
        #                 lambda _: eps_pred + eval_maxq_weight * q1_eps_pred_lora + eval_minqc_weight * q2_eps_pred_lora,
        #                 lambda _: eps_pred,
        #                 operand=None)
        eps_pred = eps_pred + eval_maxq_weight * q1_eps_pred_lora + eval_minqc_weight * q2_eps_pred_lora

        # eps_pred_pretrain = score_apply_fn({"params": score_params}, observations, current_x, input_time,
        #                             training=training,
        #                             train_lora=False)
        # eps_pred = eps_pred_lora + eps_pred_pretrain

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(actions.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), (q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis)
    
    rng, key = jax.random.split(rng, 2)
    input_tuple, outputs = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))
    q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = outputs
    # for _ in range(repeat_last_step):
    #     input_tuple, () = fn(input_tuple, 0)

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.5, 1.5)

    return action_0, rng, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis
    

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler_eval_bc_demo(actor_apply_fn, actor_params, T, rng, act_dim, actions, alphas, alpha_hats, betas,
                 sample_temperature, repeat_last_step, clip_sampler, training=False, train_lora=False):
    batch_size = actions.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis=1)
        eps_pred = actor_apply_fn({"params": actor_params}, current_x, input_time,
                                  training=training, train_lora=False)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                              shape=(actions.shape[0], current_x.shape[1]), )

        z_scaled = sample_temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        # if clip_sampler:
        #     current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng), jnp.arange(T - 1, -1, -1))

    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1.5, 1.5)

    return action_0, rng


