import os
import json
import re
from jaxrl5.agents import Pretrain, LoRASLearner, LoRALearner
from jaxrl5.data.dsrl_datasets import DSRLDataset, Toy_dataset
from ml_collections import config_flags, ConfigDict
import pickle as pkl

def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d

def load_diffusion_model(model_location, pickle, env):
    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    keys = None
    config_dict = dict(cfg['rl_config'])
    model_cls = config_dict.pop("model_cls")
    
    agent = globals()[model_cls].create(
                cfg['seed'], env.observation_space, env.action_space, **config_dict
            )

    def get_model_file(pickle):
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith(pickle):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = get_model_file(pickle=pickle)
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent

def load_ws_model(model_location, pickle, env, config_dict, model_cls):
    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    keys = None
    
    agent = globals()[model_cls].create(
                cfg['seed'], env.observation_space, env.action_space, **config_dict
            )

    def get_model_file(pickle):
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith(pickle):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = get_model_file(pickle=pickle)
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent

def load_diffusion_model_demo(model_location, pickle):
    
    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    env = '8gaussians-multitarget'
    ds = Toy_dataset(env)
    keys = None
    sample = ds.sample_jax(cfg['batch_size'], keys=keys)
    config_dict = dict(cfg['rl_config'])
    model_cls = config_dict.pop("model_cls") 

    agent = globals()[model_cls].create(
            cfg['seed'], sample['actions'][0], **config_dict
        )
    def get_model_file(pickle):
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith(pickle):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = get_model_file(pickle=pickle)
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent

def load_lora_model(seed, model_location, pickle, env, pretrain_model):

    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))
        
    with open(pretrain_model, 'rb') as file:
            cfg['rl_config']['pretrain_weights'] = pkl.load(file)
    keys = None
    config_dict = dict(cfg['rl_config'])
    model_cls = config_dict.pop("model_cls")
    cfg['seed'] = seed
    agent = globals()[model_cls].create(
                cfg['seed'], env.observation_space, env.action_space, **config_dict
            )

    def get_model_file(pickle):
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith(pickle):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = get_model_file(pickle=pickle)
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent