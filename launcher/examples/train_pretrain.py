import os
import numpy as np
from absl import app, flags
import sys
current_work_path = os.getcwd()
sys.path.insert(0, current_work_path)
from examples.states.train_diffusion_psec import call_main
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
flags.DEFINE_integer('seed', 0, 'Choose seed')
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
    info = 'Pretrain'
    timestamp = f'{info}'
    constant_parameters = dict(project='PSEC',
                               experiment_name='ddpm_lora',
                               timestamp=timestamp,
                               max_steps=1,
                               batch_size=2048,
                               eval_episodes=10,
                               log_interval=1000,
                               save_steps=250000,
                               eval_interval=250000,
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
                                   model_cls='Pretrain',
                                   actor_lr=3e-4,
                                   T=5,
                                   N=64,
                                   M=0,
                                   actor_dropout_rate=0.1,
                                   actor_num_blocks=3,
                                   decay_steps=int(3e6),
                                   actor_layer_norm=True,
                                   actor_tau=0.001,
                                   beta_schedule='vp',
                                   ),
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
        filtered_variants.append(variant)

    print(len(filtered_variants))
    variant = filtered_variants[FLAGS.variant]
    print(FLAGS.variant)

    variant['seed'] = FLAGS.seed
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
