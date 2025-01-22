"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, lora_MLP, Ensemble, StateActionValue, StateValue, DDPM, FourierFeatures,lora_FourierFeatures, cosine_beta_schedule, ddpm_sampler, ddpm_sampler_eval1,ddpm_sampler_eval_bc, MLPResNet, LoRAResNet, get_weight_decay_mask, vp_beta_schedule, ddpm_sampler_eval, LoRADense
from jax import config
# import os
# config.update("jax_debug_nans", True)
# jax.disable_jit(disable=True)[source]
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
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

class DDPMIQLLearner(Agent):
    score_model: TrainState
    target_score_model: TrainState
    lora_model: TrainState
    target_lora_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    policy_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    alpha: float
    alg: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        lora_lr: Union[float, optax.Schedule] = 1e-5,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.7,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 3,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = True,
        use_lora: bool = True,
        value_layer_norm: bool = True,
        policy_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 128,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(3e6),
        alpha: float = 0.1,
        alg: str = 'sql'
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, lora_key, critic_key, value_key = jax.random.split(rng, 5)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)
        lora_preprocess_time_cls = partial(lora_FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        lora_cond_model_cls = partial(lora_MLP,
                                hidden_dims=(time_dim * 2, time_dim * 2),
                                activations=nn.swish,
                                activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)
            # lora_lr = optax.cosine_decay_schedule(lora_lr, decay_steps)
        base_model_cls = partial(MLPResNet,
                                    use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish,
                                    use_lora=use_lora)
        
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis=0)
        actions = jnp.expand_dims(actions, axis=0)
        actor_params = actor_def.init(actor_key, observations, actions, time)['params']
        # params_pretrain = {k: v for k, v in actor_params.items() if 'lora' not in k}
        actor_optimiser = optax.adamw(learning_rate=actor_lr)
        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=actor_optimiser)
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))
        

        lora_model_cls = partial(LoRAResNet, use_layer_norm=actor_layer_norm,
                                    num_blocks=actor_num_blocks,
                                    dropout_rate=actor_dropout_rate,
                                    out_dim=action_dim,
                                    activations=nn.swish,
                                    use_lora=use_lora)
        lora_def = DDPM(time_preprocess_cls=lora_preprocess_time_cls,
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
        
        # # combine the score model and the lora model
        # # joint = combineResnet()

        # joint_model = joint(lora_model, score_model)

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

        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, 
                                 use_layer_norm=value_layer_norm,
                                 activate_final=True)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_optimiser = optax.adam(learning_rate=value_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)
        if alg == 'sql':
            alg = jnp.bool_(1)
        else:
            alg = jnp.bool_(0)
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
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            lora_model=lora_model,
            target_lora_model=target_lora_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
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
            critic_hyperparam=critic_hyperparam,
            clip_sampler=clip_sampler,
            policy_temperature=policy_temperature,
            alpha=alpha,
            alg=alg
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
            value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean()

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)
        return agent, info

    @jax.jit
    def f(agent, x):
        jax.debug.print("🤯 {x} 🤯", x=x)
        # y = jnp.sin(x)
        jax.debug.breakpoint()
        # jax.debug.print("🤯 {y} 🤯", y=y)
        return x
    def update_v_sql_eql(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch['observations'],
            batch['actions']
        )
        q = qs.min(axis=0)
        # agent.f(q)
        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch['observations'])
            if agent.alg:
                sp_term = (q - v) / (2 * agent.alpha) + 1.0
                sp_weight = jnp.where(sp_term > 0, 1., 0.)
                value_loss = (sp_weight * (sp_term ** 2) + v / agent.alpha).mean()
                # agent.f(value_loss)
            else:
                sp_term = (q - v) / agent.alpha
                sp_term = jnp.minimum(sp_term, 5.0)
                max_sp_term = jnp.max(sp_term, axis=0)
                max_sp_term = jax.lax.stop_gradient(max_sp_term)
                value_loss = (jnp.exp(sp_term - max_sp_term) + jnp.exp(-max_sp_term) * v / agent.alpha).mean()
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
    
    def update_q_sql_eql(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch['observations']
        )
        target_q = batch['rewards'] + agent.discount * batch['masks'] * next_v
        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn({"params": critic_params}, batch['observations'], batch['actions'])
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }
        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(critic.params, agent.target_critic.params, agent.tau)
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic)
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
        # add the noise to the original action to create the noisy action
        observation = batch['observations']
        # action, agent = agent.eval_actions(observation)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # use the noisy action predict the noise
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions,
                                       time,
                                       rngs={'dropout': key},
                                       training=True,
                                       train_lora=train_lora)

            # use the predict noise and the true noise to create the loss
            actor_loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return actor_loss, {'actor_loss': actor_loss}
        # get the gradients
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        # use the gradients to update the score model
        score_model = agent.score_model.apply_gradients(grads=grads)

        # replace the score model of the agent
        agent = agent.replace(score_model=score_model)
        # soft update the target score params
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )
        # replace the target score params of the target model to obtain the new target score model
        target_score_model = agent.target_score_model.replace(params=target_score_params)
        # replace the model of the agent to create the new agent
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        # return the new agent and the related information
        return new_agent, info

    def maxQ_update_actor(agent, batch: DatasetDict, train_lora: bool = True) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        # observations = batch['observations']
        
        def actor_loss_fn(lora_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:

            # Multi-samples for training
            # observations = jnp.expand_dims(batch['observations'], axis=0).repeat(4, axis = 0)
            # observations = observations.reshape(-1, observations.shape[-1])

            observations = batch['observations']
            actions, rng = ddpm_sampler(agent.lora_model.apply_fn,
                                        lora_model_params,
                                        agent.score_model.apply_fn,
                                        agent.score_model.params,
                                        agent.T,
                                        agent.rng,
                                        agent.act_dim,
                                        observations,
                                        agent.alphas,
                                        agent.alpha_hats,
                                        agent.betas,
                                        agent.ddpm_temperature,
                                        agent.M,
                                        agent.clip_sampler,
                                        rngs={'dropout': key},
                                        training=True,
                                        )

            qs = agent.critic.apply_fn(
                {"params": agent.critic.params}, observations, actions
            )
            # BC loss
            q = qs.min(axis=0)
            # actor_loss = -q.mean() /  jnp.abs(q).mean()
            actor_loss = -q.mean()

            return actor_loss, {'finetune_actor_loss': actor_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.lora_model.params)

        '''choose the lora_a and lora_b params to update'''
        # grads = {'MLPResNet_0': {'MLPResNetBlock_lora_0': {'LoRADense_0': {'lora_a': grads['MLPResNet_0']['MLPResNetBlock_lora_0']['LoRADense_0']['lora_a'],
        #                                                                 'lora_b': grads['MLPResNet_0']['MLPResNetBlock_lora_0']['LoRADense_0']['lora_b']}},
        #                           'MLPResNetBlock_lora_1': {'LoRADense_0': {'lora_a': grads['MLPResNet_0']['MLPResNetBlock_lora_1']['LoRADense_0']['lora_a'],
        #                                                                     'lora_b': grads['MLPResNet_0']['MLPResNetBlock_lora_1']['LoRADense_0']['lora_b']}},
        #                           'MLPResNetBlock_lora_2': {'LoRADense_0': {'lora_a': grads['MLPResNet_0']['MLPResNetBlock_lora_2']['LoRADense_0']['lora_a'],
        #                                                                     'lora_b': grads['MLPResNet_0']['MLPResNetBlock_lora_2']['LoRADense_0']['lora_b']}}},
        #          'MLPResNet_0': {'LoRADense_0': {'lora_a': grads['MLPResNet_0']['LoRADense_0']['lora_a'],
        #                                           'lora_b': grads['MLPResNet_0']['LoRADense_0']['lora_b']}}}
        # if train_lora:
        #     grads = {k: v for k, v in grads.items() if 'lora_a' not in k or 'lora_b' not in k}
        # JAX_TRACEBACK_FILTERING=off

        lora_model = agent.lora_model.apply_gradients(grads=grads)
        agent = agent.replace(lora_model=lora_model)

        target_lora_params = optax.incremental_update(
            lora_model.params, agent.target_lora_model.params, agent.tau
        )

        target_lora_model = agent.target_lora_model.replace(params=target_lora_params)
        new_agent = agent.replace(lora_model=lora_model, target_lora_model=target_lora_model, rng=rng)

        return new_agent, info
    def eval_actions(self, observations: jnp.ndarray, train_lora: bool = False):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        lora_params = self.target_lora_model.params

        actions, rng, eps_pred_lora_dis, eps_pred_dis = ddpm_sampler_eval(self.lora_model.apply_fn, lora_params, self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, training=False, train_lora=train_lora)
        rng, _ = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        idx = jnp.argmax(qs)
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng), eps_pred_lora_dis, eps_pred_dis
    # sample the action based on the state
    def eval_actions_bc(self, observations: jnp.ndarray, train_lora: bool = False):
        rng = self.rng

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis=0).repeat(self.N, axis = 0)

        score_params = self.target_score_model.params
        lora_params = self.target_lora_model.params

        actions, rng = ddpm_sampler_eval_bc(self.score_model.apply_fn, score_params, self.T, rng, self.act_dim, observations, self.alphas, self.alpha_hats, self.betas, self.ddpm_temperature, self.M, self.clip_sampler, training = False, train_lora = train_lora)
        rng, _ = jax.random.split(rng, 2)
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions)
        idx = jnp.argmax(qs)
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)
    @jax.jit
    def actor_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        # new_agent, actor_info = new_agent.update_actor_sql_eql(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict):
        new_agent = self
        new_agent, critic_info = new_agent.update_v(batch)
        new_agent, value_info = new_agent.update_q(batch)
        # new_agent, critic_info = new_agent.update_v_sql_eql(batch)
        # new_agent, value_info = new_agent.update_q_sql_eql(batch)

        return new_agent, {**critic_info, **value_info}

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, value_info = new_agent.update_v(batch)
        new_agent, critic_info = new_agent.update_q(batch)
        return new_agent, {**actor_info}
        # return new_agent_bc, new_agent, {**actor_info, **critic_info, **value_info}

    @jax.jit
    def update_lora(self, batch: DatasetDict):
        new_agent = self
        new_agent_bc, actor_info = new_agent.update_actor(batch, train_lora=False)
        # new_agent = new_agent_bc
        new_agent, value_info = new_agent_bc.update_v(batch)
        new_agent, critic_info = new_agent.update_q(batch)
        new_agent, fine_info = new_agent.maxQ_update_actor(batch, train_lora=True)
        # return new_agent_bc, new_agent, {**actor_info, **fine_info}
        return new_agent_bc, new_agent, {**actor_info, **critic_info, **value_info, **fine_info}
        # SQL and EQL method implement
        # new_agent, value_info = new_agent.update_v_sql_eql(batch)
        # new_agent, actor_info = new_agent.update_actor_sql_eql(batch)
        # new_agent, critic_info = new_agent.update_q_sql_eql(batch)
        # return new_agent, {**actor_info, **critic_info, **value_info}

    def choose_loss(self, q1, q2, condition):
        if condition:
            return jax.lax.cond(condition, lambda _: -q1.mean() / jnp.abs(q2).mean(), lambda _: -q2.mean() / jnp.abs(q1).mean())
        else:
            return jax.lax.cond(condition, lambda _: -q1.mean() / jnp.abs(q2).mean(), lambda _: -q2.mean() / jnp.abs(q1).mean())
