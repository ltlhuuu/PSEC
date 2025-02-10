<div align="center">
  <div style="margin-bottom: 30px">  <!-- 减少底部间距 -->
    <div style="display: flex; flex-direction: column; align-items: center; gap: 8px">  <!-- 新增垂直布局容器 -->
      <h1 align="center" style="margin: 0; line-height: 1;">
        <span style="font-size: 48px; font-weight: 600;">PSEC</span>
      </h1>
    </div>
    <h2 style="font-size: 32px; margin: 20px 0;">Skill Expansion and Composition in Parameter Space</h2>
    <h4 style="color: #666; margin-bottom: 25px;">International Conference on Learning Representation (ICLR), 2025</h4>
    <p align="center" style="margin: 30px 0;">
      <a href="https://arxiv.org/abs/2405.19909">
        <img src="https://img.shields.io/badge/arXiv-2405.19909-b31b1b.svg">
      </a>
      &nbsp;&nbsp;
      <a href="https://ltlhuuu.github.io/PSEC/">
        <img src="https://img.shields.io/badge/🌐_Project_Page-PSEC-blue.svg">
      </a>
      &nbsp;&nbsp;
      <a href="https://arxiv.org/pdf/2405.19909.pdf">
        <img src="https://img.shields.io/badge/📑_Paper-PSEC-green.svg">
      </a>
    </p>
  </div>
</div>

<div align="center">
  <p style="font-size: 20px; font-weight: 600; margin-bottom: 20px;">
    🔥 Official Implementation
  </p>
  <p style="font-size: 18px; max-width: 800px; margin: 0 auto;">
            <img src="assets/icon.svg" width="20"> <b>PSEC</b> is a novel framework designed to:
  </p>
</div>
<div align="left">
  <p style="font-size: 15px; font-weight: 600; margin-bottom: 20px;">
    🚀 <b>Facilitate</b> efficient and flexible skill expansion and composition <br>
     🔄 <b>Iteratively evolve</b> the agents' capabilities<br>
      ⚡ <b>Efficiently address</b> new challenges
  </p>
</div>

<p align="center">
 <img src="assets/intro.png" width="800" style="margin: 40px 0;">
</p>
<!-- <div align="center">
 <a href="https://github.com/ltlhuuu/PSEC/stargazers">
   <img src="https://img.shields.io/github/stars/ltlhuuu/PSEC?style=social" alt="GitHub stars">
 </a>
 &nbsp;
 <a href="https://github.com/ltlhuuu/PSEC/network/members">
   <img src="https://img.shields.io/github/forks/ltlhuuu/PSEC?style=social" alt="GitHub forks">
 </a>
 &nbsp;
 <a href="https://github.com/ltlhuuu/PSEC/issues">
   <img src="https://img.shields.io/github/issues/ltlhuuu/PSEC?style=social" alt="GitHub issues">
 </a>
</div> -->


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
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=GLWf2fq0bX}
}
```

## Acknowledgements

Parts of this code are adapted from [IDQL](https://github.com/philippe-eecs/IDQL).
