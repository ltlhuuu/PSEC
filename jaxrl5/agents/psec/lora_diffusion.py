"""Implementations of algorithms for continuous control."""
import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
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
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, DDPM, ddpm_sampler_eval,StateActionValue_demo, FourierFeatures, ddpm_sampler_eval_bc, cosine_beta_schedule, ddpm_sampler, MLPResNet, get_weight_decay_mask, vp_beta_schedule, lora_FourierFeatures, lora_MLP, LoRADense, LoRAResNet
from jaxrl5.networks.diffusion import dpm_solver_sampler_1st, vp_sde_schedule, ddpm_sampler_first_demo, ddpm_sampler_eval_demo, ddpm_sampler_eval_bc_demo, ddpm_sampler_first_demo_single, ddpm_sampler_minus_demo, ddpm_sampler_eval_lora
from jaxrl5.evaluation_dsrl import get_energy, get_reward, get_energy_jax, get_reward_jax

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values

@partial(jax.jit, static_argnames=('value_fn'))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({'params': value_params}, observations)
    return v_values

@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class MetaDrive(Agent):
    score_model: TrainState
    target_score_model: TrainState
    score_model1: TrainState
    target_score_model1: TrainState
    score_model2: TrainState
    target_score_model2: TrainState
    q_lora_model: TrainState
    q_target_lora_model: TrainState
    qc_lora_model: TrainState
    qc_target_lora_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    sampling_method: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    train_maxq_weight: float
    train_minqc_weight: float
    eval_maxq_weight: float
    eval_minqc_weight: float
    w0: float
    w1: float
    w2: float


    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        # actions: jnp.ndarray,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        use_lora: bool = True,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        actor_objective: str = 'bc',
        critic_objective: str = 'expectile',
        critic_type: str = 'hj',
        sampling_method: str = 'ddpm',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        extract_method: bool = False,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200.,
        train_maxq_weight: float = 0.07,
        train_minqc_weight: float = 0.07,
        eval_maxq_weight: float = 0.07,
        eval_minqc_weight: float = 0.07,
        w0: float = -9.0,
        w1: float = 5.0,
        w2: float = 5.0,

    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, actor_key1, actor_key2, lora_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 9)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)
        

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        q_lora_preprocess_time_cls = partial(lora_FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        q_lora_cond_model_cls = partial(lora_MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)
        
        qc_lora_preprocess_time_cls = partial(lora_FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)
        qc_lora_cond_model_cls = partial(lora_MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        # if actor_architecture == 'mlp':
        #     base_model_cls = partial(MLP,
        #                             hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
        #                             activations=mish,
        #                             use_layer_norm=actor_layer_norm,
        #                             activate_final=False,
        #                             use_lora=use_lora)
            
        #     actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
        #                      cond_encoder_cls=cond_model_cls,
        #                      reverse_encoder_cls=base_model_cls)

        # elif actor_architecture == 'ln_resnet':

        # TODO: change
        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=mish,
                                    use_lora=use_lora)
        # TODO: change
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)

        # else:
        #     raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions, time)['params']
        actor_params1 = actor_def.init(actor_key1, observations, actions, time)['params']
        actor_params2 = actor_def.init(actor_key2, observations, actions, time)['params']

        actor_optimiser = optax.adamw(learning_rate=actor_lr)

        # score_model = TrainState.create(apply_fn=actor_def.apply,
        #                                 params=actor_params,
        #                                 tx=optax.adamw(learning_rate=actor_lr, 
        #                                                weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
        #                                                mask=get_weight_decay_mask,))
        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=actor_optimiser)
        score_model1 = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params1,
                                        tx=actor_optimiser)
        score_model2 = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params2,
                                        tx=actor_optimiser)
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        target_score_model1 = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params1,
                                               tx=optax.GradientTransformation(
                                                   lambda _: None, lambda _: None
                                               ))
        target_score_model2 = TrainState.create(apply_fn=actor_def.apply,
                                                params=actor_params2,
                                                tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None
                                                ))
        
        q_lora_model_cls = partial(LoRAResNet, use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish,
                                    use_lora=use_lora)

        q_lora_def = DDPM(time_preprocess_cls=q_lora_preprocess_time_cls,
                            cond_encoder_cls=q_lora_cond_model_cls,
                            reverse_encoder_cls=q_lora_model_cls)
        
        q_lora_params = q_lora_def.init(lora_key, observations, actions, time)['params']
        q_lora_optimiser = optax.adamw(learning_rate=actor_lr)
        q_lora_model = TrainState.create(apply_fn=q_lora_def.apply, 
                                       params=q_lora_params,
                                       tx=q_lora_optimiser)
        q_target_lora_model = TrainState.create(apply_fn=q_lora_def.apply, 
                                              params=q_lora_params,
                                              tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        
        qc_lora_model_cls = partial(LoRAResNet, use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish,
                                    use_lora=use_lora)
        qc_lora_def = DDPM(time_preprocess_cls=qc_lora_preprocess_time_cls,
                            cond_encoder_cls=qc_lora_cond_model_cls,
                            reverse_encoder_cls=qc_lora_model_cls)
        qc_lora_params = qc_lora_def.init(lora_key, observations, actions, time)['params']
        qc_lora_optimiser = optax.adamw(learning_rate=actor_lr)
        qc_lora_model = TrainState.create(apply_fn=qc_lora_def.apply, 
                                       params=qc_lora_params,
                                       tx=qc_lora_optimiser)
        qc_target_lora_model = TrainState.create(apply_fn=qc_lora_def.apply, 
                                              params=qc_lora_params,
                                              tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        # bug
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)

        # no change
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if critic_type == 'qc':
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            critic_def = Ensemble(critic_cls, num=num_qs)

            # critic_cls = partial(Relu_StateActionValue, base_cls=critic_base_cls)
            # critic_def = Ensemble(critic_cls, num=num_qs)

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )


        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_optimiser = optax.adam(learning_rate=value_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)

        if critic_type == 'qc':
            value_def = StateValue(base_cls=value_base_cls)
            # value_def = Relu_StateValue(base_cls=value_base_cls)

        safe_value_params = value_def.init(safe_value_key, observations)["params"]

        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

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
            score_model1=score_model1,
            target_score_model1=target_score_model1,
            score_model2=score_model2,
            target_score_model2=target_score_model2,
            q_lora_model=q_lora_model,
            q_target_lora_model=q_target_lora_model,
            qc_lora_model=qc_lora_model,
            qc_target_lora_model=qc_target_lora_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
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
            actor_objective=actor_objective,
            sampling_method=sampling_method,
            critic_objective=critic_objective,
            critic_type=critic_type,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
            train_maxq_weight=train_maxq_weight,
            train_minqc_weight=train_minqc_weight,
            eval_maxq_weight=eval_maxq_weight,
            eval_minqc_weight=eval_minqc_weight,
            w0=w0,
            w1=w1,
            w2=w2
        )

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])

            if agent.critic_objective == 'expectile':
                value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()
            else:
                raise ValueError(f'Invalid critic objective: {agent.critic_objective}')

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)

        return agent, info
    
    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info
    
    def update_vc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)

        def safe_value_loss_fn(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vc = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])

            safe_value_loss = safe_expectile_loss(qc - vc, agent.cost_critic_hyperparam).mean()

            return safe_value_loss, {"safe_value_loss": safe_value_loss, "vc": vc.mean(), "vc_max": vc.max(), "vc_min": vc.min()}

        grads, info = jax.grad(safe_value_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)

        agent = agent.replace(safe_value=safe_value)

        return agent, info
    
    def update_qc(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )
        if agent.critic_type == "hj":
            qc_nonterminal = (1. - agent.discount) * batch["costs"] + agent.discount * jnp.maximum(batch["costs"], next_vc)
            target_qc = qc_nonterminal * batch["masks"] + batch["costs"] * (1 - batch["masks"])
        elif agent.critic_type == 'qc':
            target_qc = batch["costs"] + agent.discount * batch["masks"] * next_vc
        else:
            raise ValueError(f'Invalid critic type: {agent.critic_type}')

            
        def safe_critic_loss_fn(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qcs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params}, batch["observations"], batch["actions"]
            )
            safe_critic_loss = ((qcs - target_qc) ** 2).mean()

            return safe_critic_loss, {
                "safe_critic_loss": safe_critic_loss,
                "qc": qcs.mean(),
                "qc_max": qcs.max(),
                "qc_min": qcs.min(),
                "costs": batch["costs"].mean()
            }

        grads, info = jax.grad(safe_critic_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    def update_actor_weight(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        if agent.sampling_method == 'dpm_solver-1':
            # continuous time SDE
            eps = 1e-3
            time = jax.random.uniform(key, (batch['actions'].shape[0], )) * (1. - eps) + eps
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            alpha_t, sigma_t = vp_sde_schedule(time)
            time = jnp.expand_dims(time, axis=1)
            noisy_actions = alpha_t[:, None] * batch['actions'] + sigma_t[:, None] * noise_sample
        elif agent.sampling_method == 'ddpm':
            # discrete time SDE
            time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            
            alpha_hats = agent.alpha_hats[time]
            time = jnp.expand_dims(time, axis=1)
            alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
            alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
            noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
        else:
            raise ValueError(f'Invalid samplint method: {agent.sampling_method}')

        key, rng = jax.random.split(rng, 2)
        
        '''
        reward adv
        '''
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        '''
        cost reward
        '''
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        qc = qcs.max(axis=0)

        vc = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params}, batch["observations"]
            )

        if agent.critic_type == "qc":
            qc = qc - agent.qc_thres
            vc = vc - agent.qc_thres

        if agent.actor_objective == "feasibility":
            eps = 0. if agent.critic_type != 'qc' else 0.
            
            unsafe_condition = jnp.where(vc > 0. - eps, 1, 0)
            safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc <= 0. - eps, 1, 0)
            
            cost_exp_adv = jnp.exp((vc-qc) * agent.cost_temperature)
            reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
            
            unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub) ## ignore vc >0, qc>vc
            safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
            
            weights = unsafe_weights + safe_weights
        elif agent.actor_objective == "bc":
            weights = jnp.ones(qc.shape)
        else:
            raise ValueError(f'Invalid actor objective: {agent.actor_objective}')
        
        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * weights).mean()
            return actor_loss, {'actor_loss': actor_loss, 'weights' : weights.mean()}
            
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        agent = agent.replace(score_model=score_model)
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )
        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info
    
    def update_actor(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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
                                       train_lora=train_lora)

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
    
    def update_weight_reward(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
        observation = batch['observations']
        key, rng = jax.random.split(rng, 2)

        qs = agent.target_critic.apply_fn(
                    {"params": agent.target_critic.params},
                    batch['observations'],
                    batch["actions"],
                )

        q = qs.min(axis=0)
        v = agent.value.apply_fn(
            {'params': agent.value.params}, batch['observations']
        )
        adv = q - v
        weight = jnp.exp(adv)
        
        # add the filter to the weight
        # condition = jnp.where(adv > 0, 1.0, 0)
        # weight = weight * condition

        def actor_loss_fn(score_model1_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model1.apply_fn({'params': score_model1_params},
                                       observation,
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True,
                                       train_lora=train_lora)

            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1) * weight).mean()
            return actor_loss, {'reward_actor_loss': actor_loss, 'reward_weight': weight.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model1.params)
        score_model1 = agent.score_model1.apply_gradients(grads=grads)
        agent = agent.replace(score_model1=score_model1)
        target_score_params = optax.incremental_update(
            score_model1.params, agent.target_score_model1.params, agent.actor_tau
        )
        target_score_model1 = agent.target_score_model1.replace(params=target_score_params)
        new_agent = agent.replace(score_model1=score_model1, target_score_model1=target_score_model1, rng=rng)
        
        return new_agent, info
    
    def update_weight_cost(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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

        qcs = agent.safe_target_critic.apply_fn(
            {'params': agent.safe_target_critic.params},
            batch['observations'], batch['actions'])
        
        qc = qcs.max(axis=0)
        vc = agent.safe_value.apply_fn(
            {'params': agent.safe_value.params}, batch['observations']
        )
        cost_adv = vc - qc
        weight = jnp.exp(cost_adv)

        # add the filter to the weight
        # cost_adv = jnp.where(cost_adv < 0, 1.0, 0)
        # weight = weight * cost_adv


        def actor_loss_fn(score_model2_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # use the noisy action predict the noise
            eps_pred = agent.score_model2.apply_fn({'params': score_model2_params},
                                       observation,
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True,
                                       train_lora=train_lora)

            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1) * weight).mean()
            return actor_loss, {'cost_actor_loss': actor_loss, 'cost_weight': weight.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model2.params)
        score_model2 = agent.score_model2.apply_gradients(grads=grads)
        agent = agent.replace(score_model2=score_model2)
        target_score_params = optax.incremental_update(
            score_model2.params, agent.target_score_model2.params, agent.actor_tau
        )
        target_score_model2 = agent.target_score_model2.replace(params=target_score_params)
        new_agent = agent.replace(score_model2=score_model2, target_score_model2=target_score_model2, rng=rng)

        return new_agent, info
    
    def update_actor_energy(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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
        # add the noise to the original action to create the noisy action
#         observation = batch['observations']
        # action, agent = agent.eval_actions(observation)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["actions"],
        )

        q = qs.min(axis=0)
        # v = qs.mean(axis=0)
        v = q.mean()

        energy_exp_adv = jnp.exp((q) * 10)
        energy_weights =  jnp.clip(energy_exp_adv, 0, 100) # safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        # energy_weights_binary = (q > v)
        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # use the noisy action predict the noise
            q1_eps_pred_lora = agent.q_lora_model.apply_fn({"params": score_model_params}, 
                                            noisy_actions, 
                                            time, 
                                            rngs={'dropout': key},
                                            training=True, 
                                            train_lora=True,
                                            stop_gradient=False,
                                            weight=agent.train_maxq_weight,)
            actor_loss = (((q1_eps_pred_lora - noise_sample) ** 2).sum(axis = -1) * energy_weights).mean()
            return actor_loss, {'energy_actor_loss': actor_loss, 'energy_weights' : energy_weights.mean()}
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.q_lora_model.params)
        q_lora_model = agent.q_lora_model.apply_gradients(grads=grads)

        agent = agent.replace(q_lora_model=q_lora_model)
        q_target_lora_model_params = optax.incremental_update(
            q_lora_model.params, agent.q_target_lora_model.params, agent.actor_tau
        )
        q_target_lora_model = agent.q_target_lora_model.replace(params=q_target_lora_model_params)
        new_agent = agent.replace(q_lora_model=q_lora_model, q_target_lora_model = q_target_lora_model, rng=rng)
        return new_agent, info
    
    def update_actor_reward(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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
        # add the noise to the original action to create the noisy action
#         observation = batch['observations']
        # action, agent = agent.eval_actions(observation)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        qcs = agent.safe_critic.apply_fn(
                {"params": agent.safe_critic.params}, batch['actions']
            )
        qc = qcs.min(axis=0)
        # vc = qcs.mean(axis=0)
        vc = qcs.mean()

        # qs = agent.target_critic.apply_fn(
        #     {"params": agent.target_critic.params},
        #     batch["actions"],
        # )

        # q = qs.min(axis=0)
        # v = qs.mean(axis=0)
        reward_exp_adv = jnp.exp((qc) * 10)
        reward_weights =  jnp.clip(reward_exp_adv, 0, 100) # safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        # reward_weights_binary = (qc > vc)
        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # use the noisy action predict the noise
            # eps_pred = agent.score_model.apply_fn({'params': score_model_params},
            #                   #          batch['observations'],
            #                            noisy_actions,
            #                            time,
            #                            rngs={'dropout': key},
            #                            training=True,
            #                            train_lora=train_lora)
            q2_eps_pred_lora = agent.qc_lora_model.apply_fn({"params": score_model_params}, 
                                            noisy_actions, 
                                            time, 
                                            rngs={'dropout': key},
                                            training=True, 
                                            train_lora=True,
                                            stop_gradient=False,
                                            weight=agent.train_maxq_weight,)

            # use the predict noise and the true noise to create the loss
            # actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            actor_loss = (((q2_eps_pred_lora - noise_sample) ** 2).sum(axis = -1) * reward_weights).mean()
            return actor_loss, {'reward_actor_loss': actor_loss, 'reward_weights' : reward_weights.mean()}
        # get the gradients
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.qc_lora_model.params)
        # use the gradients to update the score model
        qc_lora_model = agent.qc_lora_model.apply_gradients(grads=grads)

        # replace the score model of the agent
        agent = agent.replace(qc_lora_model=qc_lora_model)
        # soft update the target score params
        qc_target_lora_model_params = optax.incremental_update(
            qc_lora_model.params, agent.qc_target_lora_model.params, agent.actor_tau
        )
        # replace the target score params of the target model to obtain the new target score model
        qc_target_lora_model = agent.qc_target_lora_model.replace(params=qc_target_lora_model_params)
        # replace the model of the agent to create the new agent
        new_agent = agent.replace(qc_lora_model=qc_lora_model, qc_target_lora_model=qc_target_lora_model, rng=rng)

        # return the new agent and the related information
        return new_agent, info
    def update_actor_energy_weight(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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
        # add the noise to the original action to create the noisy action
#         observation = batch['observations']
        # action, agent = agent.eval_actions(observation)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
         # safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        # energy_weights_binary = (q > v)
        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
            {"params": score_model_params}, # agent.target_critic.params
            batch["actions"],
            )

            q = qs.min(axis=0)
            # v = qs.mean(axis=0)
            v = q.mean()

            energy_exp_adv = jnp.exp((q) * 5.0)
            energy_weights =  jnp.clip(energy_exp_adv, 0, 100)
            # use the noisy action predict the noise
            eps_pred = agent.score_model.apply_fn({'params': agent.score_model.params},
                              #          batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=False,
                                       train_lora=train_lora)
            # q1_eps_pred_lora = agent.q_lora_model.apply_fn({"params": score_model_params}, 
            #                                 noisy_actions, 
            #                                 time, 
            #                                 rngs={'dropout': key},
            #                                 training=True, 
            #                                 train_lora=True,
            #                                 stop_gradient=False,
            #                                 weight=agent.train_maxq_weight,)

            # use the predict noise and the true noise to create the loss
            # actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * energy_weights).mean()
            return actor_loss, {'energy_actor_loss': actor_loss, 'energy_weights' : energy_weights.mean()}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info
    
    def update_actor_reward_weight(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
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
        # add the noise to the original action to create the noisy action
#         observation = batch['observations']
        # action, agent = agent.eval_actions(observation)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)
        # safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        # reward_weights_binary = (qc > vc)
        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # use the noisy action predict the noise
            qcs = agent.safe_critic.apply_fn(
                {"params": score_model_params}, batch['actions']
            )
            qc = qcs.min(axis=0)
            # vc = qcs.mean(axis=0)
            vc = qcs.mean()

            # qs = agent.target_critic.apply_fn(
            #     {"params": agent.target_critic.params},
            #     batch["actions"],
            # )

            # q = qs.min(axis=0)
            # v = qs.mean(axis=0)
            reward_exp_adv = jnp.exp((qc) * 5.0)
            reward_weights =  jnp.clip(reward_exp_adv, 0, 100) 
            eps_pred = agent.score_model.apply_fn({'params': agent.score_model.params},
                              #          batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=False,
                                       train_lora=train_lora)

            # use the predict noise and the true noise to create the loss
            # actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * reward_weights).mean()
            return actor_loss, {'reward_actor_loss': actor_loss, 'reward_weights' : reward_weights.mean()}
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.safe_critic.params)
        
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)

        agent = agent.replace(safe_critic=safe_critic)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info

    def maxQ_update_actor(agent, batch: DatasetDict, train_lora: bool = True) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(lora_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
          #   observations = batch['observations']
          action = batch['actions']
          actions, rng = ddpm_sampler_minus_demo(agent.q_lora_model.apply_fn,
                                        lora_model_params,
                                        agent.score_model.apply_fn,
                                        agent.score_model.params,
                                        agent.T,
                                        agent.rng,
                                        agent.act_dim,
                                        action,
                                        agent.alphas,
                                        agent.alpha_hats,
                                        agent.betas,
                                        agent.ddpm_temperature,
                                        agent.M,
                                        agent.clip_sampler,
                                        rngs={'dropout': key},
                                        training=True,
                                        weight=agent.train_maxq_weight,
                                        )
          qs = agent.critic.apply_fn(
          {"params": agent.critic.params}, actions
          )
          q = qs.min(axis=0)

          actor_loss = -q.mean()
          return actor_loss, {'q_actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.q_lora_model.params)
        q_lora_model = agent.q_lora_model.apply_gradients(grads=grads)
        agent = agent.replace(q_lora_model=q_lora_model)

        q_target_lora_params = optax.incremental_update(
            q_lora_model.params, agent.q_target_lora_model.params, agent.actor_tau
        )
        q_target_lora_model = agent.q_target_lora_model.replace(params=q_target_lora_params)
        new_agent = agent.replace(q_lora_model=q_lora_model, q_target_lora_model=q_target_lora_model)
        return new_agent, info
    
    def minQC_update_actor(agent, batch: DatasetDict, train_lora: bool = True) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(lora_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            action = batch['actions']
            actions, rng = ddpm_sampler_minus_demo(agent.qc_lora_model.apply_fn,
                                        lora_model_params,
                                        agent.score_model.apply_fn,
                                        agent.score_model.params,
                                        agent.T,
                                        agent.rng,
                                        agent.act_dim,
                                        action,
                                        agent.alphas,
                                        agent.alpha_hats,
                                        agent.betas,
                                        agent.ddpm_temperature,
                                        agent.M,
                                        agent.clip_sampler,
                                        rngs={'dropout': key},
                                        training=True,
                                        weight=agent.train_minqc_weight,
                                        )

            qcs = agent.safe_critic.apply_fn(
                {"params": agent.safe_critic.params}, actions
            )
            qc = qcs.min(axis=0)

            actor_loss = -qc.mean()
            return actor_loss, {'qc_actor_loss': actor_loss}
        
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.qc_lora_model.params)
        qc_lora_model = agent.qc_lora_model.apply_gradients(grads=grads)
        agent = agent.replace(qc_lora_model=qc_lora_model)

        qc_target_lora_params = optax.incremental_update(
            qc_lora_model.params, agent.qc_target_lora_model.params, agent.actor_tau
        )
        qc_target_lora_model = agent.qc_target_lora_model.replace(params=qc_target_lora_params)
        new_agent = agent.replace(qc_lora_model=qc_lora_model, qc_target_lora_model=qc_target_lora_model)
        return new_agent, info

    def lora_update_actor(agent, batch: DatasetDict, train_lora: bool = False) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        if agent.sampling_method == 'dpm_solver-1':
            # continuous time SDE
            eps = 1e-3
            time = jax.random.uniform(key, (batch['actions'].shape[0], )) * (1. - eps) + eps
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            alpha_t, sigma_t = vp_sde_schedule(time)
            time = jnp.expand_dims(time, axis=1)
            noisy_actions = alpha_t[:, None] * batch['actions'] + sigma_t[:, None] * noise_sample
        elif agent.sampling_method == 'ddpm':
            # discrete time SDE
            time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
            key, rng = jax.random.split(rng, 2)
            noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
            
            alpha_hats = agent.alpha_hats[time]
            time = jnp.expand_dims(time, axis=1)
            alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
            alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
            noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
        else:
            raise ValueError(f'Invalid samplint method: {agent.sampling_method}')

        key, rng = jax.random.split(rng, 2)
        
        '''
        reward adv
        '''
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
          #   batch["observations"],
            batch["actions"],
        )

        q = qs.min(axis=0)

        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        
        # '''
        # cost reward
        # '''
        # qcs = agent.safe_target_critic.apply_fn(
        #     {"params": agent.safe_target_critic.params},
        #     batch["observations"],
        #     batch["actions"],
        # )

        # qc = qcs.max(axis=0)

        # vc = agent.safe_value.apply_fn(
        #         {"params": agent.safe_value.params}, batch["observations"]
        #     )

        # if agent.critic_type == "qc":
        #     qc = qc - agent.qc_thres
        #     vc = vc - agent.qc_thres


        if train_lora:
            reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
            weights = jnp.clip(reward_exp_adv, 0, 100)
        else:
            weights = jnp.ones(q.shape)

        
        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True,
                                       train_lora=train_lora)
            
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis = -1) * weights).mean()
            
            if train_lora:
                actor_info = {'lora_actor_loss': actor_loss, 'lora_weights': weights.mean()}
            else:
                actor_info = {'actor_loss': actor_loss, 'weights': weights.mean()}

            return actor_loss, actor_info
            
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        
        agent = agent.replace(score_model=score_model)

        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )

        target_score_model = agent.target_score_model.replace(params=target_score_params)

        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    def eval_actions(self, actions: jnp.ndarray, train_lora: bool = False, eval_maxq_weight: float = 0.1, eval_minqc_weight: float = 0.1):
        rng = self.rng

#         assert len(observations.shape) == 1
        actions = jax.device_put(actions)
        action = jnp.expand_dims(actions, axis = 0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        q_lora_params = self.q_target_lora_model.params
        qc_lora_params = self.qc_target_lora_model.params
        
        if self.sampling_method == 'ddpm':
            actions, rng, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = ddpm_sampler_eval_demo(self.q_lora_model.apply_fn,q_lora_params,
                                                                                                       self.qc_lora_model.apply_fn,qc_lora_params,
                                                                                                       self.score_model.apply_fn, score_params, 
                                                                                                       self.T, 
                                                                                                       rng, 
                                                                                                       self.act_dim, 
                                                                                                       action, 
                                                                                                       self.alphas, 
                                                                                                       self.alpha_hats, 
                                                                                                       self.betas,
                                                                                                       self.ddpm_temperature,
                                                                                                       self.M, 
                                                                                                       self.clip_sampler, 
                                                                                                       training=False, 
                                                                                                       train_lora=train_lora, 
                                                                                                       train_maxq_weight=self.train_maxq_weight,
                                                                                                       train_minqc_weight=self.train_minqc_weight,
                                                                                                       eval_maxq_weight=eval_maxq_weight,
                                                                                                       eval_minqc_weight=eval_minqc_weight)
        elif self.sampling_method == 'dpm_solver-1':
            actions, rng = dpm_solver_sampler_1st(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler)
        else:
            raise ValueError(f'Invalid sampling method: {self.sampling_method}')
        
        rng, key = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, actions)
        sum = qs + qcs
        idx = jnp.argmax(sum)
        action = actions[idx]
        new_rng = rng

        return np.array(action.squeeze()), self.replace(rng=new_rng), q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis
    
    def eval_actions_bc(self, observations: jnp.ndarray, train_lora: bool = False):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        # lora_params = self.target_lora_model.params

        actions, rng = ddpm_sampler_eval_bc(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, training = False, train_lora = train_lora)
        rng, _ = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions)

        # idx = jnp.argmax(qs) # maxq
        idx = jnp.argmin(qcs)  # minqc

        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    
    def eval_actions_lora(self, observations: jnp.ndarray, train_lora: bool = False):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        score_params1 = self.target_score_model1.params
        score_params2 = self.target_score_model2.params
        # lora_params = self.target_lora_model.params

        actions, rng = ddpm_sampler_eval_lora(self.score_model.apply_fn, score_params, 
                                              self.score_model1.apply_fn, score_params1, 
                                              self.score_model2.apply_fn, score_params2,
                                               self.T, rng, self.act_dim, observations, 
                                               self.alphas, self.alpha_hats, self.betas, 
                                               self.ddpm_temperature, self.M, self.clip_sampler, 
                                               training = False, train_lora = train_lora, w0=self.w0, w1=self.w1, w2=self.w2)

        rng, _ = jax.random.split(rng, 2)

        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions)
        
        # idx = jnp.argmax(qs) # maxq
        idx = jnp.argmin(qcs) # minqc

        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)

    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info
    
    @jax.jit
    def eval_loss(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.actor_loss_no_grad(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict):
        def slice(x):
            return x[:256]

        new_agent = self
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)

        return new_agent, {**critic_info, **value_info}

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = int(batch['observations'].shape[0]/2)

        def first_half(x):
            return x[:batch_size]
        
        def second_half(x):
            return x[batch_size:]
        
        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)

        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)

        def slice(x):
            return x[:256]
        
        mini_batch = jax.tree_util.tree_map(slice, batch)
        new_agent, critic_info = new_agent.update_v(mini_batch)
        new_agent, value_info = new_agent.update_q(mini_batch)
        new_agent, safe_critic_info = new_agent.update_vc(mini_batch)
        new_agent, safe_value_info = new_agent.update_qc(mini_batch)

        return new_agent, {**actor_info, **critic_info, **value_info, **safe_critic_info, **safe_value_info}

    @jax.jit
    def update_lora(self, batch: DatasetDict):
        # new_agent = agent_bc
        new_agent = self


        # new_agent, actor_info = new_agent.update_actor(batch, train_lora=False)

        # def slice(x):
        #     return x[:256]
        # mini_batch = jax.tree_util.tree_map(slice, batch)
        # new_agent, critic_info = new_agent.update_v(batch)
        new_agent, value_info = new_agent.update_q(batch)
        new_agent, q_lora_actor_info = new_agent.update_actor_energy_weight(batch, train_lora=True)

        # new_agent, safe_critic_info = new_agent_bc.update_vc(batch)
        new_agent, safe_value_info = new_agent.update_qc(batch)
        new_agent, qc_lora_actor_info = new_agent.update_actor_reward_weight(batch, train_lora=True)

        return  new_agent, {**q_lora_actor_info, **qc_lora_actor_info, **value_info, **safe_value_info} 
    
    @jax.jit
    def update_bc(self, batch: DatasetDict):
        new_agent = self

        new_agent, actor_info = new_agent.update_actor(batch, train_lora=False)

        return new_agent, actor_info
    
    @jax.jit
    def update_lora2tasks(self, batch: DatasetDict):
        new_agent = self
        
        new_agent, energy_actor_info = new_agent.update_weight_reward(batch, train_lora=True) # train_lora does not work
        new_agent, critic_info = new_agent.update_v(batch)
        new_agent, value_info = new_agent.update_q(batch)

        new_agent, reward_actor_info = new_agent.update_weight_cost(batch, train_lora=True)
        new_agent, safe_critic_info = new_agent.update_vc(batch)
        new_agent, safe_value_info = new_agent.update_qc(batch)

        return new_agent, {**energy_actor_info, **reward_actor_info, **value_info, **critic_info, **safe_value_info, **safe_critic_info}

    
    @jax.jit
    def update_weight_bc(self, batch: DatasetDict):
        new_agent = self
        
        new_agent, value_info = new_agent.update_q(batch)
        new_agent, actor_info = new_agent.update_weight_energy(batch, train_lora=False)

        return new_agent, {**actor_info, **value_info}
    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent# 
    