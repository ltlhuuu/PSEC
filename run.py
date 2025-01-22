import subprocess

num_variants = 9


seeds = [42]


for i in range(num_variants):
        for seed in seeds:

            command = f"python launcher/examples/train_pretrain.py --variant {i} --seed {seed}"
            subprocess.run(command, shell=True)
            
            command = f"python launcher/examples/train_lora_finetune.py --variant {i} --model_cls 'LoRALearner' --seed {seed}"
            subprocess.run(command, shell=True)

            command = f"python launcher/examples/train_lora_finetune.py --variant {i} --model_cls 'LoRASLearner' --seed {seed}"
            subprocess.run(command, shell=True)


