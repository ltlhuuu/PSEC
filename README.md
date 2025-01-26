# PSEC: Skill Expansion and Composition in Parameter Space
International Conference on Learning Representation (ICLR), 2025 

Paper Link : <a href="https://arxiv.org/abs/2405.19909">
  <img src="https://img.shields.io/badge/arXiv-2405.19909-<COLOR>.svg" alt="arXiv" style="vertical-align: middle;">
</a>


🔥 The official implementation of PSEC, a new framework
designed to **facilitate efficient and flexible skill expansion and composition, iteratively evolve the agents’ capabilities and  efficiently address new challenges**. 

<p float="left">
<img src="assets/intro.png" width="800">
</p>

## Quick start
Clone this repository and navigate to PSEC folder
```python
git clone https://github.com/ltlhuuu/PSEC.git
cd PSEC
```
## Environment Installation
Environment configuration and dependencies are available in environment.yaml and requirements.txt.

Create conda environment for this experiments
```python
conda create -n PSEC python=3.9
conda activate PSEC
```
Then install the remaining requirements (with MuJoCo already downloaded, if not see [here](#MuJoCo-installation)): 
```bash
pip install -r requirements.txt
```

Install the `MetaDrive` environment via
```python
pip install git+https://github.com/HenryLHH/metadrive_clean.git@main
```
### MuJoCo installation
Download MuJoCo:
```bash
mkdir ~/.mujoco
cd ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
cd mujoco210
wget https://www.roboti.us/file/mjkey.txt
```
Then add the following line to `.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```
## Run experiments
### Pretrain
Pretrain the model with the following command. Meanwhile there are pre-trained models, you can download them from [here](https://drive.google.com/drive/folders/1lpcShmYoKVt4YMH66JBiA0MhYEV9aEYy?usp=sharing).
```python
export XLA_PYTHON_CLIENT_PREALLOCATE=False
CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_pretrain.py --variant 0 --seed 0
```
### LoRA finetune
Train the skill policies with LoRA to achieve skill expansion. Meanwhile there are pre-trained models, you can download them from [here](https://drive.google.com/drive/folders/1lpcShmYoKVt4YMH66JBiA0MhYEV9aEYy?usp=sharing).
```python
CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_lora_finetune.py --com_method 0 --model_cls 'LoRALearner' --variant 0 --seed 0
```
### Context-aware Composition
Train the context-aware modular to adaptively leverage different skill knowledge to solve the tasks. You can download the pretrained model and datasets from [here](https://drive.google.com/drive/folders/1lpcShmYoKVt4YMH66JBiA0MhYEV9aEYy?usp=sharing). Then, run the following command,
```python
CUDA_VISIBLE_DEVICES=0 python launcher/examples/train_lora_finetune.py --com_method 0 --model_cls 'LoRASLearner' --variant 0 --seed 0
```

## Citations
If you find our paper and code useful for your research, please cite:
```
@inproceedings{
liu2025psec,
title={Skill Expansion and Composition in Parameter Space},
author={Tenglong Liu, Jianxiong Li, Yinan Zheng, Haoyi Niu, Yixing Lan, Xin Xu, Xianyuan Zhan},
booktitle={International Conference on Learning Representations},
year={2025},
url={https://openreview.net/XXX}
}
```

## Acknowledgements

Parts of this code are adapted from [IDQL](https://github.com/philippe-eecs/IDQL).