import subprocess
import os

seeds = [0]
variant = 0
gpu_id = 1  # Specify which GPU to use (0, 1, 2, etc.)
dataset_paths = [
                'dynamic_dataset/source/hc_0.25f.pkl',
                'dynamic_dataset/source/hc_2g.pkl',
                'dynamic_dataset/source/hc_2t.pkl',
                ]
                
for seed in seeds:
    for dataset_path in dataset_paths:
        env_vars = {
            'CUDA_VISIBLE_DEVICES': str(gpu_id),
            **dict(os.environ)
        }
        command = f"python launcher/examples/train_ddpm_psec.py --variant {variant} --seed {seed} --dataset_path {dataset_path}"
        subprocess.run(command, shell=True, env=env_vars)
