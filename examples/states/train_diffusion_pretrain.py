import gym
import jax
import wandb
from tqdm import tqdm
from flax.core import frozen_dict
from jaxrl5.agents import BCLearner, IQLLearner, PretrainLearner, MLPWBCLearner
from jaxrl5.data.d4rl_datasets import D4RLDataset, DynamicDataset
from jaxrl5.evaluation import evaluate, implicit_evaluate
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo
from jaxrl5.data import ReplayBuffer, BinaryDataset
import jax.numpy as jnp
import numpy as np
from tensorboardX import SummaryWriter
import os

@jax.jit
def merge_batch(batch1, batch2):
    merge = {}
    for k in batch1.keys():
        merge[k] = jnp.concatenate([batch1[k], batch2[k]], axis = 0)
    
    return frozen_dict.freeze(merge)

def call_main(details):
    wandb.init(project=details['project'], name=details['group'])
    wandb.config.update(details)
    env = gym.make(details['env_name'])
    ds = DynamicDataset(dataset_path=details['dataset_path'], env=env)
<<<<<<< HEAD:examples/states/train_diffusion_offline_BC.py
    
=======
>>>>>>> 34c4ca25690fda2ddb19eeb5fee6e511533a9371:examples/states/train_diffusion_pretrain.py
    
    env = wrap_gym(env)
    if details['save_video']:
        env = WANDBVideo(env)

    if details['take_top'] is not None or details['filter_threshold'] is not None:
        ds.filter(take_top=details['take_top'], threshold=details['filter_threshold'])

    config_dict = details['rl_config']

    model_cls = config_dict.pop("model_cls")

    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    keys = None
<<<<<<< HEAD:examples/states/train_diffusion_offline_BC.py
=======

>>>>>>> 34c4ca25690fda2ddb19eeb5fee6e511533a9371:examples/states/train_diffusion_pretrain.py
    sample = ds.sample_jax(details['batch_size'], keys=keys)
    log_path = f"log/halfcheetah/{details['timestamp']}_{details['group']}/{details['experiment_name']}_bc"
    writer = SummaryWriter(log_path)
    eval_result = []
    eval_path = os.path.join(log_path, "evaluation_results.npy")
    for i in tqdm(range(details['max_steps']), smoothing=0.1):
        sample = ds.sample_jax(details['batch_size'], keys=keys)
        agent, info = agent.update(sample)
        wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
        if i % details['save_model_interval'] == 0:
            agent.save(f"./results/{details['timestamp']}_{details['group']}/{details['experiment_name']}_bc", i)
        
        if i % details['eval_interval'] == 0 and i > 0:
            for inference_params in details['inference_variants']:
                agent = agent.replace(**inference_params)
                eval_info = evaluate(
                    agent, env, details['eval_episodes'], save_video=details['save_video']
                )
                eval_result.append(eval_info['return'])
                if 'binary' not in details['env_name']:
                    eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
                wandb.log({f"eval/{inference_params}_{k}": v for k, v in eval_info.items()}, step=i)
                for k, v in eval_info.items():
                    writer.add_scalar(f"eval/{inference_params}_{k}", v, i)          
            agent.replace(**details['training_time_inference_params'])
    np.save(eval_path, eval_result)