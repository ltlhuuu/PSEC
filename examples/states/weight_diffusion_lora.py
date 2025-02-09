import gymnasium as gym
import jax
import wandb
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.agents import LoRALearner, LoRASLearner
from jaxrl5.data.dsrl_datasets import DSRLDataset, Toy_dataset
from jaxrl5.evaluation_dsrl import evaluate_lora_reward, evaluate_lora_cost, evaluate_composition, evaluate_score_com, evaluate_action_com
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
import os
from jax import config
import time
from examples.load_model import load_diffusion_model, load_lora_model
import pickle
from env.utils import get_imitation_data
from tensorboardX import SummaryWriter

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

    if 'LoRALearner' in details['rl_config']['model_cls'] and details['com_method'] == 0:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['rl_config']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['rl_config']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
        
    else:
        env = gym.make(details['env_name'])
        ws_data = get_imitation_data(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['rl_config']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], data_location=ws_data)

        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['rl_config']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)



    ds.seed(details["seed"])

    if details['save_video']:
        env = WANDBVideo(env)

    config_dict = details['rl_config']
    model_cls = config_dict.pop("model_cls")
    
    if 'LoRA' in model_cls:
        details['pretrain_model'] = f"results/Pretrain/Pretrain_{details['env_name']}_ddpm_iql_/ddpm_iql_bc/model1000000.pickle"
        with open(details['pretrain_model'], 'rb') as file:
            config_dict['pretrain_weights'] = pickle.load(file)
            
    if model_cls == 'LoRASLearner':
        info1 = f"rank{details['rl_config']['rank']}-alpha{details['rl_config']['alpha_r']}-reward{details['rl_config']['reward_temperature']}-cost{details['rl_config']['cost_temperature']}"
        lora0_model = f"results/LoRA-{info1}/LoRA-{info1}_{details['env_name']}_ddpm_lora_/ddpm_lora/model1.pickle"
        with open(lora0_model, 'rb') as lora_file:
             config_dict['lora0'] = pickle.load(lora_file)
             
        LoRA_pickle = f"model1.pickle"
        model_location = f"results/LoRA-{info1}/LoRA-{info1}_{details['env_name']}_ddpm_lora_/ddpm_lora"
        env, agent_lora = load_lora_model(seed=details['seed'], model_location=model_location, pickle=LoRA_pickle, env=env, pretrain_model=details['pretrain_model'])
    elif (model_cls == 'LoRALearner' and details['com_method'] == 1) or (model_cls == 'LoRALearner' and details['com_method'] == 2):
        info1 = f"rank{details['rl_config']['rank']}-alpha{details['rl_config']['alpha_r']}-reward{details['rl_config']['reward_temperature']}-cost{details['rl_config']['cost_temperature']}"
        LoRA_pickle = f"model100000.pickle"
        model_location = f"results/LoRA-{info1}/LoRA-{info1}_{details['env_name']}_ddpm_lora_/ddpm_lora"
        env, agent_lora = load_lora_model(seed=details['seed'], model_location=model_location, pickle=LoRA_pickle, env=env, pretrain_model=details['pretrain_model'])
    keys = None
    sample = ds.sample_jax(details['batch_size'], keys=keys)
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
    writer = SummaryWriter(f"log/{details['timestamp']}_{details['env_name']}_{details['seed']}")
    
    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        
        if model_cls == 'LoRALearner' and details['com_method'] == 0:
            agent, info = agent.update_lora(sample)
        elif model_cls == 'LoRALearner' and details['com_method'] == 1:
            agent, info = agent.update_lora_score_com(agent_lora, sample) # score level composition to train the ws
        elif model_cls == 'LoRALearner' and details['com_method'] == 2:
            agent, info = agent.update_lora_action_com(agent_lora, sample) # action level composition to train the ws
        elif model_cls == 'LoRASLearner':
            agent, info = agent.update_composition(agent_lora, sample)
        else:
            agent, info = agent.update_lora2tasks(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train_bc/{k}": v for k, v in info.items()}, step=i)
            for k, v in info.items():
                writer.add_scalar(f"train_bc/{k}", v, i)
        if i % details['save_steps'] == 0:
            agent.save(f"./results/{details['timestamp']}/{details['timestamp']}_{details['group']}/{details['experiment_name']}", i)
        cost_score = 10
        if i % details['eval_interval'] == 0 and i > 0:
            for inference_params in details['inference_variants']:
                agent = agent.replace(**inference_params)
                
                if model_cls == 'LoRASLearner':
                    eval_info = evaluate_composition(agent, agent_lora, env, details['eval_episodes'])
                    eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
                    wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i)
                elif model_cls == 'LoRALearner' and details['com_method'] == 0:
                    reward_eval_info = evaluate_lora_reward(agent, env, details['eval_episodes'])
                    reward_eval_info['q_normalized_return'], reward_eval_info['q_normalized_cost'] = env.get_normalized_score(reward_eval_info['q_return'], reward_eval_info['q_cost'])
                    wandb.log({f"eval/{inference_params}_{k}": v for k, v in reward_eval_info.items()}, step=i)
                    cost_eval_info = evaluate_lora_cost(agent, env, details['eval_episodes'])
                    cost_eval_info['qc_normalized_return'], cost_eval_info['qc_normalized_cost'] = env.get_normalized_score(cost_eval_info['qc_return'], cost_eval_info['qc_cost'])
                    wandb.log({f"eval/{inference_params}_{k}": v for k, v in cost_eval_info.items()}, step=i)
                    # for k, v in reward_eval_info.items():
                    #     writer.add_scalar(f"eval_bc/{inference_params}_{k}", v, i)
                    # for k, v in cost_eval_info.items():
                    #     writer.add_scalar(f"eval_bc/{inference_params}_{k}", v, i)
                    if cost_eval_info['qc_normalized_cost'] <= cost_score:
                        cost_score = cost_eval_info['qc_normalized_cost']
                        agent.save(f"./results/{details['timestamp']}/{details['timestamp']}_{details['group']}/{details['experiment_name']}", "1")
                elif model_cls == 'LoRALearner' and details['com_method'] == 1:
                    eval_info = evaluate_score_com(agent, agent_lora, env, details['eval_episodes'])
                    eval_info['normalized_return'], eval_info['normalized_cost'] = env.get_normalized_score(eval_info['return'], eval_info['cost'])
                    wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i)
                    # for k, v in eval_info.items():
                    #     writer.add_scalar(f"eval_bc/{inference_params}_{k}", v, i)
                elif model_cls == 'LoRALearner' and details['com_method'] == 2:
                    eval_info = evaluate_action_com(agent, agent_lora, env, details['eval_episodes'])
                    eval_info['normalized_return'], eval_info['normalized_cost'] = env.get_normalized_score(eval_info['return'], eval_info['cost'])
                    wandb.log({f"eval_bc/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i)
                    # for k, v in eval_info.items():
                    #     writer.add_scalar(f"eval_bc/{inference_params}_{k}", v, i)
                
            agent.replace(**details['training_time_inference_params'])
