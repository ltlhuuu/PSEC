import subprocess

num_variants = 8
seeds = [0]

gpu_id = 0 

for i in range(num_variants):
    for seed in seeds:
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python launcher/examples/train_pretrain.py --variant {i} --seed {seed}"
        subprocess.run(command, shell=True)
