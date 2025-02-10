<!-- <div align="center">
  <div id="user-content-toc" style="margin-bottom: 50px">
    <ul align="center" style="list-style: none; padding: 0;">
      <summary>
        <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
          <img src="assets/icon.svg" width="50" style="margin: 0;">
          <b>PSEC</b>
        </h1>
        <h1>Skill Expansion and Composition in Parameter Space</h1>
        <h3>International Conference on Learning Representation (ICLR), 2025</h3>
        <h2><a href="https://arxiv.org/abs/2405.19909">Paper</a> &emsp; <a href="https://ltlhuuu.github.io/PSEC/">Project page</a></h2>
      </summary>
    </ul>
  </div>
</div>

🔥 The official implementation of PSEC, a new framework
designed to **facilitate efficient and flexible skill expansion and composition, iteratively evolve the agents’ capabilities and  efficiently address new challenges**. 

<p float="left">
<img src="assets/intro.png" width="800">
</p>

Paper Link : <a href="https://arxiv.org/abs/2405.19909">
  <img src="https://img.shields.io/badge/arXiv-2405.19909-<COLOR>.svg" alt="arXiv" style="vertical-align: middle;">
</a>

Project Page : <a href="https://ltlhuuu.github.io/PSEC/">
  <img src="https://img.shields.io/badge/Project-PSEC-<COLOR>.svg" alt="Project Page" style="vertical-align: middle;">
</a> -->

<div align="center">
 <div id="user-content-toc" style="margin-bottom: 50px">
   <ul align="center" style="list-style: none; padding: 0;">
     <summary>
       <h1 style="display: flex; align-items: center; justify-content: center; gap: 15px;">
         <img src="assets/icon.svg" width="60" style="margin: 0;">
         <span style="font-size: 48px; font-weight: 600;">PSEC</span>
       </h1>
       <h1 style="font-size: 32px; margin: 20px 0;">Skill Expansion and Composition in Parameter Space</h1>
       <h3 style="color: #666; margin-bottom: 25px;">International Conference on Learning Representation (ICLR), 2025</h3>
       <p align="center" style="margin: 30px 0;">
         <a href="https://arxiv.org/abs/2405.19909">
           <img src="https://img.shields.io/badge/arXiv-2405.19909-b31b1b.svg" height="20">
         </a>
         &nbsp;&nbsp;
         <a href="https://ltlhuuu.github.io/PSEC/">
           <img src="https://img.shields.io/badge/🌐_Project_Page-PSEC-blue.svg" height="20">
         </a>
         &nbsp;&nbsp;
         <a href="https://arxiv.org/pdf/2405.19909.pdf">
           <img src="https://img.shields.io/badge/📑_Paper-PSEC-green.svg" height="20">
         </a>
       </p>
     </summary>
   </ul>
 </div>
</div>

<div align="center">
 <div style="background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,236,219,0.5) 50%, rgba(255,255,255,0) 100%); padding: 25px 0;">
   <p style="font-size: 24px; font-weight: 600; margin-bottom: 25px;">
     🔥 Official PyTorch Implementation
   </p>
 </div>

 <div style="max-width: 850px; margin: 0 auto; padding: 20px;">
   <p style="font-size: 20px; margin-bottom: 30px;">
     <b>PSEC</b> is a novel framework designed to:
   </p>
   
   <ul style="list-style: none; padding: 0; text-align: left; font-size: 18px; line-height: 1.6;">
     <li style="margin: 15px 0; padding: 10px 20px; background: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
       🚀 <b>Facilitate</b> efficient and flexible skill expansion and composition
     </li>
     <li style="margin: 15px 0; padding: 10px 20px; background: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
       🔄 <b>Iteratively evolve</b> the agents' capabilities
     </li>
     <li style="margin: 15px 0; padding: 10px 20px; background: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
       ⚡ <b>Efficiently address</b> new challenges
     </li>
   </ul>
 </div>
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
