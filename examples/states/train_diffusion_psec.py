import gym
import jax
import wandb
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.agents import Pretrain
from jaxrl5.data.dsrl_datasets import DSRLDataset, Toy_dataset
from jaxrl5.evaluation_dsrl import evaluate_bc
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
from jax import config
import time
from dataset.utils import get_imitation_data

timestamp = int(time.time())

@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for k in batch1.keys():
        merge[k] = jnp.concatenate([batch1[k], batch2[k]], axis = 0)
    
    return frozen_dict.freeze(merge)

def call_main(details):
    wandb.init(project=details['project'], name=details['group'])
    wandb.config.update(details)

    if details['env_name'] == '8gaussians-multitarget':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = details['env_name']
        ds = Toy_dataset(env)

    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env)
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)

    ds.seed(details["seed"])

    if details['save_video']:
        env = WANDBVideo(env)

    config_dict = details['rl_config']
    model_cls = config_dict.pop("model_cls")

    if "BC" in model_cls:
        agent_bc = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = ["observations", "actions"]
    else:
        
        agent_bc = globals()[model_cls].create(
            details['seed'], env.observation_space, env.action_space, **config_dict
        )
        keys = None

    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        agent_bc, info_bc = agent_bc.update_bc(sample)
        
        if i % details['log_interval'] == 0:
            wandb.log({f"train_bc/{k}": v for k, v in info_bc.items()}, step=i)

        if i % details['save_steps'] == 0:
            agent_bc.save(f"./results/{details['timestamp']}/{details['timestamp']}_{details['group']}/{details['experiment_name']}_bc", i)

        if i % details['eval_interval'] == 0 and i > 0:
            for inference_params in details['inference_variants']:
                agent_bc = agent_bc.replace(**inference_params)
                eval_info_bc = evaluate_bc(agent_bc, env, details['eval_episodes'], train_lora=False)
                eval_info_bc["normalized_return"], eval_info_bc["normalized_cost"] = env.get_normalized_score(eval_info_bc["return"], eval_info_bc["cost"])
                wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info_bc.items()}, step=i)
            agent_bc.replace(**details['training_time_inference_params'])
