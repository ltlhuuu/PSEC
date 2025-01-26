import os
import numpy as np
from absl import app, flags
import pickle
import sys

currerent_path = os.getcwd()
sys.path.insert(0, currerent_path)

from examples.states.train_diffusion_lora import call_main
from launcher.hyperparameters import set_hyperparameters
from jax import config
from ml_collections import config_flags, ConfigDict
import json
import time
from datetime import datetime

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_integer('seed', 2, 'Choose seed')
flags.DEFINE_integer('trajs', 10000, 'trajs number')
flags.DEFINE_integer('rank', 8, 'LoRA rank')
flags.DEFINE_integer('alpha_r', 16, 'LoRA alpha')
flags.DEFINE_integer('train_ws', 0, 'Train the ws')
flags.DEFINE_string('model_cls', 'LoRALearner', 'Model name: LoRALearner, ComLoRALearner')
flags.DEFINE_string('pretrain_model', 'results/dynamic_dataset/source/hc_2g_halfcheetah_ddpm_iql_/ddpm_iql_bc/model200000.pickle', 'pretrain model path')
flags.DEFINE_string('info', '2g', 'File path to the training hyperparameter configuration.')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def main(_):
    # FLAGS.info = FLAGS.pretrain_model.split('/')[-3].split('_')[1]
    print('train_ws:',FLAGS.train_ws)
    timestamp = f'{FLAGS.info}'
    constant_parameters = dict(project='offline_schedule_final',
                               experiment_name='LoRA',
                               task_id=0,
                                timestamp=timestamp,
                               max_steps=200001, 
                               pretrain_model=FLAGS.pretrain_model,
                               batch_size=512,
                               eval_episodes=10,
                               log_interval=1000,
                               eval_interval=10000,
                               save_interval=100000,
                               save_video=False,
                               filter_threshold=None,
                               take_top=None,
                               online_max_steps=0,
                               unsquash_actions=False,
                               normalize_returns=True,
                               training_time_inference_params=dict(
                                N=64,
                                clip_sampler=True,
                                M=1,),
                               rl_config=dict(
                                   model_cls=FLAGS.model_cls,
                                   train_ws=FLAGS.train_ws, 
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1,
                                   actor_num_blocks=3,
                                   decay_steps=int(4e5),
                                   actor_layer_norm=True,
                                   value_layer_norm=True,
                                   actor_tau=0.001,
                                   actor_objective="exp_adv", 
                                   actor_architecture='ln_resnet',
                                   beta_schedule='vp',
                                   policy_temperature=10.0,
                                   ))

    sweep_parameters = dict(
                            seed=list(range(1)),
                            env_name=[
                                'halfcheetah-medium-v2',
                                'halfcheetah-medium-replay-v2',
                                'halfcheetah-medium-expert-v2',  
                                'walker2d-medium-v2',
                                'walker2d-medium-replay-v2', 
                                'walker2d-medium-expert-v2',  
                                      ],
                            )

    variants = [constant_parameters]
    name_keys = ['experiment_name', 'env_name']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    inference_sweep_parameters = dict(
                            N=[1],
                            clip_sampler=[True],
                            M=[0],
                            )
    
    inference_variants = [{}]
    name_keys = []
    inference_variants = set_hyperparameters(inference_sweep_parameters, inference_variants)

    filtered_variants = []
    for variant in variants:

        variant['inference_variants'] = inference_variants
            
        if 'antmaze' in variant['env_name']:
            variant['rl_config']['critic_hyperparam'] = 0.9
        else:
            variant['rl_config']['critic_hyperparam'] = 0.7

        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]

    variant['seed'] = FLAGS.seed
    variant['trajs'] = FLAGS.trajs
    variant['rl_config']['rank'] = FLAGS.rank
    variant['rl_config']['alpha_r'] = FLAGS.alpha_r
    variant['group'] = f"{variant['group']}_rank_{FLAGS.rank}_alpha_r_{FLAGS.alpha_r}"
    
    if FLAGS.model_cls == 'LoRALearner' and variant['rl_config']['train_ws']==1:
        variant['group'] = f"ws_{variant['group']}"
        variant['max_steps'] = 1001
        variant['eval_interval'] = 100
        variant['log_interval'] = 100
    if FLAGS.model_cls == 'ComLoRALearner':
        variant['timestamp'] = variant['timestamp'].replace('A-LoRA', 'Com')
        variant['max_steps'] = 100001
        variant['eval_interval'] = 10000
        variant['log_interval'] = 10000
            
    print(FLAGS.variant)
    if not os.path.exists(f"./results/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}"):
        os.makedirs(f"./results/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}")
    if not os.path.exists(f"./results/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc"):
        os.makedirs(f"./results/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc")
        
    with open(f"./results/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(variant), f, indent=4)
    with open(f"./results/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc/config.json", "w") as f:
        json.dump(to_dict(variant), f, indent=4)
    call_main(variant)


if __name__ == '__main__':
    app.run(main)
