import os
import numpy as np
from absl import app, flags
curenent_work_path = os.getcwd()
import sys
sys.path.insert(0, curenent_work_path)
from examples.states.weight_diffusion_lora import call_main
from launcher.hyperparameters import set_hyperparameters
from jax import config
from ml_collections import config_flags, ConfigDict
import json
import time

# for debug
# config.update('jax_disable_jit', True)
os.environ["WANDB_MODE"] = "offline"

FLAGS = flags.FLAGS
flags.DEFINE_integer('variant', 0, 'Logging interval.')
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_integer('seed', 56, 'Choose seed')
flags.DEFINE_integer('rank', 8, 'LoRA rank')
flags.DEFINE_integer('alpha_r', 16, 'LoRA alpha')
flags.DEFINE_integer('com_method', 0, 'psec:0, score level: 1, action level: 2')
flags.DEFINE_string('model_cls', 'LoRASLearner', 'Choose model from [LoRALearner, LoRASLearner, MetaDrive]')
flags.DEFINE_string('critic_type', 'hj', 'hj, qc, composition')
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
    info = f'LoRA-rank{FLAGS.rank}-alpha{FLAGS.alpha_r}'
    timestamp = f'{info}'
    constant_parameters = dict(project='PSEC',
                               experiment_name='ddpm_lora',
                               timestamp=timestamp,
                               max_steps=1,
                               pretrain_model='',
                               lora0='',
                               com_method=FLAGS.com_method,
                               batch_size=2048,
                               eval_episodes=10,
                               log_interval=1000,
                               save_steps=50000,
                               eval_interval=50000,
                               save_video=False,
                               filter_threshold=None,
                               take_top=None,
                               online_max_steps=0,
                               unsquash_actions=False,
                               normalize_returns=True,
                               ratio=1.0,
                               training_time_inference_params=dict(
                                N=64,
                                clip_sampler=True,
                                M=1,),
                               rl_config=dict(
                                   model_cls=FLAGS.model_cls,
                                   actor_lr=3e-4,
                                   critic_lr=3e-4,
                                   value_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1,
                                   actor_num_blocks=3,
                                   decay_steps=int(1e6),
                                   actor_layer_norm=True,
                                   value_layer_norm=True,
                                   actor_tau=0.001,
                                   critic_objective='expectile',
                                   critic_hyperparam = 0.9,
                                   cost_critic_hyperparam = 0.9,
                                   critic_type=FLAGS.critic_type, #[hj, qc]
                                   cost_ub=100,
                                   beta_schedule='vp',
                                   cost_temperature=1,
                                   reward_temperature=1,
                                   cost_limit=10,
                                   actor_objective="bc", 
                                   sampling_method="ddpm", 
                                   extract_method="minqc", 
                                   rank=FLAGS.rank,
                                   alpha_r=FLAGS.alpha_r
                                   ),
                               dataset_kwargs=dict(
                                                    cost_scale=25,
                                                    pr_data='data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
                                                )
                               )

    sweep_parameters = dict(seed=[42],
                            env_name=[
                                    "OfflineMetadrive-easysparse-v0",        # 0
                                    "OfflineMetadrive-easymean-v0",          # 1
                                    "OfflineMetadrive-easydense-v0",         # 2
                                    "OfflineMetadrive-mediumsparse-v0",      # 3
                                    "OfflineMetadrive-mediummean-v0",        # 4
                                    "OfflineMetadrive-mediumdense-v0",       # 5
                                    "OfflineMetadrive-hardsparse-v0",        # 6
                                    "OfflineMetadrive-hardmean-v0",          # 7
                                    "OfflineMetadrive-harddense-v0"          # 8
                                    ]
                            )

    variants = [constant_parameters]
    name_keys = ['env_name','experiment_name']
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
    variant['timestamp'] = f"{variant['timestamp']}-reward{variant['rl_config']['reward_temperature']}-cost{variant['rl_config']['cost_temperature']}"
    if FLAGS.model_cls == 'LoRASLearner':
        # info = info.replace('LoRA', 'ComLoRA')
        variant['timestamp'] = variant['timestamp'].replace('LoRA', 'ComLoRA')
        variant['batchsize'] = 512
        variant['max_steps'] = 1001
        variant['save_steps'] = 100
        variant['eval_interval'] = 100
        variant['log_interval'] = 100
        variant['rl_config']['value_layer_norm'] = False
    if FLAGS.critic_type == 'composition' and (FLAGS.com_method == 1 or FLAGS.com_method == 2):
        variant['timestamp'] = variant['timestamp'].replace('LoRA', 'ComLoRA-score')
        variant['batchsize'] = 512
        variant['max_steps'] = 1001
        variant['save_steps'] = 100
        variant['eval_interval'] = 100
        variant['log_interval'] = 100
        variant['rl_config']['value_layer_norm'] = False
            
    print(FLAGS.variant)
    info = variant['timestamp']
    if not os.path.exists(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}"):
        os.makedirs(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}")
    if not os.path.exists(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc"):
        os.makedirs(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc")
    with open(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(variant), f, indent=4)
    with open(f"./results/{info}/{variant['timestamp']}_{variant['group']}/{variant['experiment_name']}_bc/config.json", "w") as f:
        json.dump(to_dict(variant), f, indent=4)
    call_main(variant)


if __name__ == '__main__':
    app.run(main)
