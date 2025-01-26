import gym
import jax
import wandb
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.agents import LORALearner
from jaxrl5.data.d4rl_datasets import D4RLDataset
from jaxrl5.evaluation import evaluate_composition, evaluate_lora
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
import os
from jax import config
import time
from pathlib import Path
from datetime import datetime
import pickle
from examples.load_model import load_lora_model

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

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

    # work_dir = Path.cwd()
    work_dir = os.getcwd()
    time_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # set env
    env = gym.make(details['env_name'])
    if "binary" in details['env_name']:
        ds = BinaryDataset(env)
    else:
        ds = D4RLDataset(env)
    
    if details['save_video']:
        env = WANDBVideo(env)
    env = wrap_gym(env)
    
    # load pretrain model
    with open(details['pretrain_model'], 'rb') as file:
        details['rl_config']['pretrain_weights'] = pickle.load(file)

    config_dict = details['rl_config']
    model_cls = config_dict.pop("model_cls")
    keys = None
    
    # load lora model to train ws
    if details['rl_config']['train_ws']==1:
        pickle1 = f"model{details['trajs']}.pickle"
        model_location = f"results/A-LoRA-{details['seed']}_LoRA_{details['env_name']}_trajs_{details['trajs']}_rank_8_alpha_r_16/LoRA_bc"
        env, agent_lora = load_lora_model(seed=details['seed'], model_location=model_location, pickle=pickle1, env=env)

    
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
        if "antmaze" in details['env_name']:
            ds.dataset_dict["rewards"] -= 1.0
        elif (
            "halfcheetah" in details['env_name']
            or "walker2d" in details['env_name']
            or "hopper" in details['env_name']
        ) and details['normalize_returns']:
            ds.normalize_returns()
            
    ds, ds_val = ds.split(0.95)        
    sample = ds.sample_jax(details['batch_size'], keys=keys)
    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        if details['rl_config']['train_ws']==1:
            agent_bc, info_bc = agent_bc.update_composition(agent_lora, sample)
        else:
            agent_bc, info_bc = agent_bc.update_lora(sample)
        
        if i % details['log_interval'] == 0:
            val_sample = ds_val.sample(details['batch_size'], keys=keys)
            
            if details['rl_config']['train_ws']==1:
                agent_bc, val_info = agent_bc.update_composition(agent_lora, val_sample)
            else:
                agent_bc, val_info = agent_bc.update_lora(sample)
            wandb.log({f"train_bc/{k}": v for k, v in info_bc.items()}, step=i)
            wandb.log({f"val/{k}": v for k, v in val_info.items()}, step=i)
            
        if i % details['eval_interval'] == 0:
            agent_bc.save(f"./results/{details['timestamp']}_{details['group']}/{details['experiment_name']}_bc", i)

        if i % details['eval_interval'] == 0 and i > 0:
            for inference_params in details['inference_variants']:
                agent_bc = agent_bc.replace(**inference_params)
                if details['rl_config']['train_ws']==1:
                    eval_info = evaluate_composition(agent_bc, agent_lora, env, details['eval_episodes'], save_video=details['save_video'])
                else:
                    eval_info = evaluate_lora(agent_bc, env, details['eval_episodes'], save_video=details['save_video'])

                if 'binary' not in details['env_name']:
                    eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
                wandb.log({f"eval/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i)     
            agent_bc.replace(**details['training_time_inference_params'])
