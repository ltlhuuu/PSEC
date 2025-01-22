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
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue_ws, DDPM_alpha, ddpm_sampler_eval_bc, cosine_beta_schedule, vp_beta_schedule, ddpm_sampler_eval_alphas,StateValue
from jaxrl5.networks.lora_alpha import LoRA_FourierFeatures, LoRA_MLP, LoRAResNet

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

class LoRASLearner(Agent):
    lora_model: TrainState
    target_lora_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    value1: TrainState
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
    pretrain_weights: dict
    lora0: dict
    rank: int
    alpha_r: int


    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        pretrain_weights: dict,
        lora0: dict,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
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
        rank: int = 8,
        alpha_r: int = 16,

    ):

        rng = jax.random.PRNGKey(seed)
        rng, lora_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        lora_preprocess_time_cls = partial(LoRA_FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True,
                                      pretrain_weights=pretrain_weights)

        lora_cond_model_cls = partial(LoRA_MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False,
                                pretrain_weights=pretrain_weights,
                                lora0=lora0,
                                rank=rank,
                                alpha_r=alpha_r)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis = 0)

        
        lora_model_cls = partial(LoRAResNet, use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish,
                                    pretrain_weights=pretrain_weights,
                                    lora0=lora0,
                                    rank=rank,
                                    alpha_r=alpha_r,
                                    )

        lora_def = DDPM_alpha(time_preprocess_cls=lora_preprocess_time_cls,
                            cond_encoder_cls=lora_cond_model_cls,
                            reverse_encoder_cls=lora_model_cls)
        
        lora_params = lora_def.init(lora_key, observations, actions, time)['params']
        lora_optimiser = optax.adamw(learning_rate=actor_lr)
        lora_model = TrainState.create(apply_fn=lora_def.apply, 
                                       params=lora_params,
                                       tx=lora_optimiser)
        target_lora_model = TrainState.create(apply_fn=lora_def.apply, 
                                              params=lora_params,
                                              tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))


        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)

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


        value1_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value1_def = StateValue_ws(base_cls=value1_base_cls)
        value1_params = value1_def.init(value_key, observations)["params"]
        value1_optimiser = optax.adam(learning_rate=value_lr)

        value1 = TrainState.create(apply_fn=value1_def.apply,
                                  params=value1_params,
                                  tx=value1_optimiser)

        if critic_type == 'qc':
            value_def = StateValue(base_cls=value_base_cls)

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
            lora_model=lora_model,
            target_lora_model=target_lora_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            value1=value1,
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
            pretrain_weights=pretrain_weights,
            lora0=lora0,
            rank=rank,
            alpha_r=alpha_r
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
    
    def update_lora_composition(agent, agent_lora, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        time = jax.random.randint(key, (batch['actions'].shape[0], ), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (batch['actions'].shape[0], agent.act_dim))
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        '''
        reward adv
        '''
        # qs = agent.target_critic.apply_fn(
        #     {"params": agent.target_critic.params},
        #     batch["observations"],
        #     batch["actions"],
        # )

        # q = qs.min(axis=0)

        # v = agent.value.apply_fn(
        #     {"params": agent.value.params}, batch["observations"]
        # )
        
        # '''
        # cost reward
        # '''
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        # qc = qcs.max(axis=0)

        # vc = agent.safe_value.apply_fn(
        #         {"params": agent.safe_value.params}, batch["observations"]
        #     )

        # # if agent.critic_type == "qc":
        # #     qc = qc - agent.qc_thres
        # #     vc = vc - agent.qc_thres

        # eps = 0. if agent.critic_type != 'qc' else 0.
        
        # unsafe_condition = jnp.where(vc > 0. - eps, 1, 0)
        # safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc <= 0. - eps, 1, 0)
        
        # cost_exp_adv = jnp.exp((vc-qc) * agent.cost_temperature)
        # reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
        
        # unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub) ## ignore vc >0, qc>vc
        # safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        
        # weights = unsafe_weights + safe_weights
        qc = qcs.max(axis=0)
        vc = agent.safe_value.apply_fn(
            {'params': agent.safe_value.params}, batch['observations']
        )
        cost_adv = vc - qc
        exp_cost_adv = jnp.exp(cost_adv * agent.cost_temperature)
        mask = jnp.where((qc-vc)< 0., 1, 0)
        weights = mask * jnp.clip(exp_cost_adv, 0, agent.cost_ub)


        def composition_actor_loss_fn(value_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            lora_params = agent_lora.qc_target_lora_model.params
            alpha_as = agent.value1.apply_fn({'params': value_model_params}, batch['observations'])
            eps_pred_lora = agent.lora_model.apply_fn({"params": lora_params}, 
                                                           batch['observations'],
                                                            noisy_actions, 
                                                            time, 
                                                            alpha_as=alpha_as,
                                                            # rngs={'dropout': key},
                                                            # training=True, 
                                                            )
            # alpha_loss = (((alpha_as.mean() - 8) ** 2)).mean()
            # alpha_loss = (((alpha_as.mean() - 8) ** 2)).mean() + (((alpha_as[:,1].mean() - 16) ** 2)).mean()
            # alpha_loss = (((alpha_as[:,1].mean() - 16) ** 2)).mean()
            # y1 = alpha_as[:,0].mean()
            # y2 = alpha_as[:,1].mean()
            # loss1 = ((y1 - 16) ** 2).mean() + ((y2 - 0) ** 2).mean()
            # loss2 = ((y2 - 16) ** 2).mean() + ((y1 - 0) ** 2).mean()
            # alpha_loss = jnp.minimum(loss1, loss2)
            # actor_loss = (((eps_pred_lora - noise_sample) ** 2).sum(axis = -1)).mean() + alpha_loss
            actor_loss = (((eps_pred_lora - noise_sample) ** 2).sum(axis = -1)* weights).mean()
            return actor_loss, {'composition_actor_loss': actor_loss, 'alphas': alpha_as.mean(), 'alphas0': alpha_as[:,0][:,None].mean(), 'alphas1': alpha_as[:,1][:,None].mean()}
        
        grads, info = jax.grad(composition_actor_loss_fn, has_aux=True)(agent.value1.params)
        value1 = agent.value1.apply_gradients(grads=grads)
        agent = agent.replace(value1=value1)

        return agent, info
    
    def update_lora_composition_final(agent, agent_lora, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)


        def composition_actor_loss_fn(value_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            
            lora_params = agent_lora.qc_target_lora_model.params
            alpha_as = agent.value.apply_fn({'params': value_model_params}, batch['observations'])
            actions, rng = ddpm_sampler_eval_alphas(
                                            agent.lora_model.apply_fn, lora_params, alpha_as,
                                            agent.T, agent.rng, agent.act_dim, batch['observations'], 
                                            agent.alphas, agent.alpha_hats, agent.betas, 
                                            agent.ddpm_temperature, agent.M, agent.clip_sampler, 
                                            training = False)
            
            actor_loss = (((actions - batch['actions']) ** 2).sum(axis = -1) ).mean()
            return actor_loss, {'composition_actor_loss': actor_loss, 'alphas': alpha_as.mean(), 'alphas0': alpha_as[:,0][:,None].mean(), 'alphas1': alpha_as[:,1][:,None].mean()}
        
        grads, info = jax.grad(composition_actor_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)

        return agent, info

    
    def eval_actions_composition(self, agent_lora, observations: jnp.ndarray):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        lora_params = agent_lora.qc_target_lora_model.params
        alpha_as = self.value1.apply_fn({'params': self.value1.params}, observations)

        # save alpha_as
        actions, rng = ddpm_sampler_eval_alphas(
                                            self.lora_model.apply_fn, lora_params, alpha_as,
                                            self.T, rng, self.act_dim, observations, 
                                            self.alphas, self.alpha_hats, self.betas, 
                                            self.ddpm_temperature, self.M, self.clip_sampler, 
                                            training = False)
        rng, _ = jax.random.split(rng, 2)

        # qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations, actions)
        qcs = compute_safe_q(agent_lora.safe_target_critic.apply_fn, agent_lora.safe_target_critic.params, observations, actions)
        idx = jnp.argmin(qcs)  # minqc
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    

    @jax.jit
    def update_composition(self, agent_lora, batch: DatasetDict):
        new_agent = self
        
        new_agent, composition_actor_info = new_agent.update_lora_composition(agent_lora, batch) # train_lora does not work

        return new_agent, {**composition_actor_info}

    
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent# 
    