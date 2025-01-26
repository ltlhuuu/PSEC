import os
import sys
path = os.getcwd()
sys.path.insert(0, path)
import numpy as np
from absl import app, flags

from examples.states.train_diffusion_pretrain import call_main
from launcher.hyperparameters import set_hyperparameters
import json
from ml_collections import config_flags, ConfigDict
from datetime import datetime
from jax import config


FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')
flags.DEFINE_integer('seed', 0, 'Choose seed')
flags.DEFINE_string('dataset_path', 'dynamic_dataset/source/wk_2g.pkl', 'the path of pretrain policy')


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def main(_):
    info = FLAGS.dataset_path.split('.pkl')[0]
    timestamp = f'{info}'
    constant_parameters = dict(project='offline_schedule_final',
                               experiment_name='ddpm_iql',
                               dataset_path=FLAGS.dataset_path,
                               timestamp=timestamp,
                               max_steps=200001, #Actor takes two steps per critic step
                               save_model_interval=100000,
                               batch_size=512,
                               eval_episodes=10,
                               log_interval=1000,
                               eval_interval=50000,
                               save_video = False,
                               filter_threshold=None,
                               take_top=None,
                               online_max_steps = 0,
                               unsquash_actions=False,
                               normalize_returns=True,
                               training_time_inference_params=dict(
                               N = 64,
                               clip_sampler = True,
                               M = 0,),
                               rl_config=dict(
                                   model_cls='PretrainLearner',
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1,
                                   actor_num_blocks=3,
                                   decay_steps=int(3e6),
                                   actor_layer_norm=True,
                                   value_layer_norm=True,
                                   actor_tau=0.001,
                                   beta_schedule='vp',
                               ))

    sweep_parameters = dict(
                            seed=list(range(1)),
                            env_name=[
                                'halfcheetah-medium-v2', 'walker2d-medium-v2',
                            ],
                            )
    

    variants = [constant_parameters]
    name_keys = ['experiment_name']
    # name_keys = ['experiment_name', 'env_name']
    variants = set_hyperparameters(sweep_parameters, variants, name_keys)

    inference_sweep_parameters = dict(
                            N = [1],
                            clip_sampler = [True], 
                            M = [0],
                            )
    
    inference_variants = [{}]
    name_keys = []
    inference_variants = set_hyperparameters(inference_sweep_parameters, inference_variants)

    filtered_variants = []
    for variant in variants:
        # variant['rl_config']['T'] = variant['T']
        # variant['rl_config']['beta_schedule'] = variant['beta_schedule']
        variant['inference_variants'] = inference_variants
            
        if 'antmaze' in variant['env_name']:
            variant['rl_config']['critic_hyperparam'] = 0.9
        else:
            variant['rl_config']['critic_hyperparam'] = 0.7

        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    variant['seed'] = FLAGS.seed
    print(FLAGS.variant)
    task_id = variant['env_name'].split('-')[0]
    variant['timestamp'] = f"{timestamp}_{task_id}" 
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
