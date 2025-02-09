import gym
import jax
import wandb
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.agents import BCLearner, IQLLearner, DDPMIQLLearner, FISOR
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation_dsrl import evaluate, evaluate_pr, evaluate_bc, evaluate1, evaluate2
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
import os
from jax import config
# config.update("jax_debug_nans", True)
os.environ['WANDB_MODE'] = 'dryrun'
# wandb.disabled = True
@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for k in batch1.keys():
        merge[k] = jnp.concatenate([batch1[k], batch2[k]], axis = 0)
    
    return frozen_dict.freeze(merge)

def call_main(details):
    wandb.init(project=details['project'], name=details['group'])
    wandb.config.update(details)

    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['rl_config']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['rl_config']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['rl_config']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    if details['save_video']:
        env = WANDBVideo(env)

    if details['take_top'] is not None or details['filter_threshold'] is not None:
        ds.filter(take_top=details['take_top'], threshold=details['filter_threshold'])

    config_dict = details['rl_config']

    model_cls = config_dict.pop("model_cls")

    if "BC" in model_cls:
        agent = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = ["observations", "actions"]
    else:
        agent = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = None

    sample = ds.sample_jax(details['batch_size'], keys=keys)

    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        if details['rl_config']['use_lora']:
            agent_bc, agent, info = agent.update_lora(sample)
        else:
            agent, info = agent.update(sample)
        
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)


        if i % details['eval_interval'] == 0 and i > 0:
            for inference_params in details['inference_variants']:
                agent = agent.replace(**inference_params)
                if details['env_name'] == 'PointRobot':
                    eval_info = evaluate_pr(agent, env, details['eval_episodes'])
                else:
                    eval_info = evaluate(agent, env, details['eval_episodes'], train_lora=True,eval_maxq_weight=config_dict['eval_maxq_weight'], eval_minqc_weight=config_dict['eval_minqc_weight'])
                    eval_info1 = evaluate(agent, env, details['eval_episodes'], train_lora=True, eval_maxq_weight=config_dict['eval1_maxq_weight'], eval_minqc_weight=config_dict['eval1_minqc_weight'])
                    eval_info2 = evaluate(agent, env, details['eval_episodes'], train_lora=True, eval_maxq_weight=config_dict['eval2_maxq_weight'], eval_minqc_weight=config_dict['eval2_minqc_weight'])
                    eval_info3 = evaluate(agent, env, details['eval_episodes'], train_lora=True, eval_maxq_weight=config_dict['eval3_maxq_weight'], eval_minqc_weight=config_dict['eval3_minqc_weight'])
                    eval_info4 = evaluate(agent, env, details['eval_episodes'], train_lora=True, eval_maxq_weight=config_dict['eval4_maxq_weight'], eval_minqc_weight=config_dict['eval4_minqc_weight'])
                    eval_info5 = evaluate(agent, env, details['eval_episodes'], train_lora=True, eval_maxq_weight=config_dict['eval5_maxq_weight'], eval_minqc_weight=config_dict['eval5_minqc_weight'])
                    eval_info6 = evaluate(agent, env, details['eval_episodes'], train_lora=True, eval_maxq_weight=config_dict['eval6_maxq_weight'], eval_minqc_weight=config_dict['eval6_minqc_weight'])
                    eval_info_bc = evaluate_bc(agent_bc, env, details['eval_episodes'], train_lora=False)
                if details['env_name'] != 'PointRobot':
                    eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
                    eval_info1["normalized_return"], eval_info1["normalized_cost"] = env.get_normalized_score(eval_info1["return"], eval_info1["cost"])
                    eval_info2["normalized_return"], eval_info2["normalized_cost"] = env.get_normalized_score(eval_info2["return"], eval_info2["cost"])
                    eval_info3["normalized_return"], eval_info3["normalized_cost"] = env.get_normalized_score(eval_info3["return"], eval_info3["cost"])
                    eval_info4["normalized_return"], eval_info4["normalized_cost"] = env.get_normalized_score(eval_info4["return"], eval_info4["cost"])
                    eval_info5["normalized_return"], eval_info5["normalized_cost"] = env.get_normalized_score(eval_info5["return"], eval_info5["cost"])
                    eval_info6["normalized_return"], eval_info6["normalized_cost"] = env.get_normalized_score(eval_info6["return"], eval_info6["cost"])
                    eval_info_bc["normalized_return"], eval_info_bc["normalized_cost"] = env.get_normalized_score(eval_info_bc["return"], eval_info_bc["cost"])
                wandb.log({f"eval/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i)
                wandb.log({f"eval1/{inference_params}_{k}": v for k, v in eval_info1.items()}, step=i)
                wandb.log({f"eval2/{inference_params}_{k}": v for k, v in eval_info2.items()}, step=i)
                wandb.log({f"eval3/{inference_params}_{k}": v for k, v in eval_info3.items()}, step=i)
                wandb.log({f"eval4/{inference_params}_{k}": v for k, v in eval_info4.items()}, step=i)
                wandb.log({f"eval5/{inference_params}_{k}": v for k, v in eval_info5.items()}, step=i)
                wandb.log({f"eval6/{inference_params}_{k}": v for k, v in eval_info6.items()}, step=i)
                wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info_bc.items()}, step=i)
            agent.replace(**details['training_time_inference_params'])
            # BC agent evaluate
            # for inference_params in details['inference_variants']:
            #     agent_bc = agent_bc.replace(**inference_params)
            #     # eval_info_bc = evaluate_bc(
            #     #     agent_bc, env, details['eval_episodes'], save_video=details['save_video'],  train_lora=False
            #     # )
            #     if details['env_name'] == 'PointRobot':
            #         eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            #     else:
            #         eval_info = evaluate(agent, env, details['eval_episodes'])
            #     if details['env_name'] != 'PointRobot':
            #         eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            #     wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info_bc.items()}, step=i)
            # agent_bc.replace(**details['training_time_inference_params'])