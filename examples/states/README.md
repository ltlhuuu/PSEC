## Online

### SAC
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --config=configs/sac_config.py \
                --notqdm
```
### DroQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 300000 \
                --config=configs/droq_config.py \
                --notqdm
```
### RedQ
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=Hopper-v4 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 300000 \
                --config=configs/redq_config.py \
                --notqdm
```

## Offline

###
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_offline.py --env_name=halfcheetah-expert-v2 \
                --config=configs/bc_config.py
```

### train LoRA
```python
python launcher/examples/train_metadrive_wBC.py --variant 0 --model_cls 'LoRALearner' --com_method 0 --seed 1024 --composition 'hj'
```
### score composition and action composition with LoRA
```python
# score composition
python launcher/examples/train_metadrive_wBC.py --variant 0 --model_cls 'LoRALearner' --com_method 1 --seed 1024 --composition 'composition'

# action composition
python launcher/examples/train_metadrive_wBC.py --variant 0 --model_cls 'LoRALearner' --com_method 2 --seed 1024 --composition 'composition'
```

## parameter composition with LoRA
```python
python launcher/examples/train_metadrive_wBC.py --variant 0 --model_cls 'LoRASLearner' --com_method 0 --seed 1024 --composition 'hj'
```