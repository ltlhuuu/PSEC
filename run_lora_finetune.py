import subprocess

seeds = [0]
num_variants = 3

pretrain_models = [ 
                   'results/dynamic_dataset/source/hc_2g_halfcheetah_ddpm_iql_/ddpm_iql_bc/model200000.pickle',
                   'results/dynamic_dataset/source/hc_0.25f_halfcheetah_ddpm_iql_/ddpm_iql_bc/model200000.pickle',
                   'results/dynamic_dataset/source/hc_2t_halfcheetah_ddpm_iql_/ddpm_iql_bc/model200000.pickle',
                   'results/dynamic_dataset/source/wk_2g_walker2d_ddpm_iql_/ddpm_iql_bc/model200000.pickle',
                   'results/dynamic_dataset/source/wk_0.5f_walker2d_ddpm_iql_/ddpm_iql_bc/model200000.pickle',
                   'results/dynamic_dataset/source/wk_2t_walker2d_ddpm_iql_/ddpm_iql_bc/model200000.pickle'
                   ]

for pretrain_model in pretrain_models:
    info = pretrain_model.split('/')[-3].split('_')[1]
    if 'hc' in pretrain_model:
        for variant in range(num_variants):  
            for seed in seeds:
                command = f"CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_lora.py --variant {variant} --seed {seed} --train_ws 0 --pretrain_model {pretrain_model} --info {info}"
                subprocess.run(command, shell=True)
    else:
        for variant in range(num_variants):  
            for seed in seeds:
                command = f"CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_lora.py --variant {3+variant} --seed {seed} --train_ws 0 --pretrain_model {pretrain_model} --info {info}"
                subprocess.run(command, shell=True)