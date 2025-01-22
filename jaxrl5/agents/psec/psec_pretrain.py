"""Implementations of algorithms for continuous control."""
import os
from functools import partial
from typing import Dict, Optional, Tuple, Union
import flax.linen as nn
import gym
import gym.spaces
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, DDPM, FourierFeatures, ddpm_sampler_eval_bc, cosine_beta_schedule, MLPResNet, vp_beta_schedule



def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class Pretrain(Agent):
    score_model: TrainState
    target_score_model: TrainState
    discount: float
    tau: float
    actor_tau: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        ddpm_temperature: float = 1.0,
        actor_num_blocks: int = 2,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)
        

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # TODO: change
        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=mish,
                                    )
        # TODO: change
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions, time)['params']

        actor_optimiser = optax.adamw(learning_rate=actor_lr)

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=actor_optimiser)
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            target_score_model=target_score_model,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            clip_sampler=clip_sampler,
        )
    
    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        observation = batch['observations']
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       observation,
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True,
                                       )

            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return actor_loss, {'actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        agent = agent.replace(score_model=score_model)
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )
        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)
        
        return new_agent, info
    
    def eval_actions_bc(self, observations: jnp.ndarray, train_lora: bool = False):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params

        actions, rng = ddpm_sampler_eval_bc(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, training = False)
        rng, _ = jax.random.split(rng, 2)
        idx = 0

        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    @jax.jit
    def update_bc(self, batch: DatasetDict):
        new_agent = self

        new_agent, actor_info = new_agent.update_actor(batch)

        return new_agent, actor_info
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent# 
    