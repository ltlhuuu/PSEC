from typing import Dict

import gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa
import matplotlib.pyplot as plt
from gym.wrappers import RecordVideo
import metadrive
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from metadrive import MetaDriveEnv
import matplotlib as mpl


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True, eval_maxq_weight: float = 0.07, eval_minqc_weight: float = 0.07
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = agent.eval_actions(obs, train_lora=train_lora, eval_maxq_weight=eval_maxq_weight, eval_minqc_weight=eval_minqc_weight)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs), 'q1_eps_pred_lora_dis': np.mean(q1_eps_pred_lora_dis), 'q2_eps_pred_lora_dis': np.mean(q2_eps_pred_lora_dis), 'eps_pred_dis': np.mean(eps_pred_dis)}

def evaluate1(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True, eval_maxq_weight: float = 0.07, eval_minqc_weight: float = 0.07
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = agent.eval_actions(obs, train_lora=train_lora, eval_maxq_weight=eval_maxq_weight, eval_minqc_weight=eval_minqc_weight)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs), 'q1_eps_pred_lora_dis': np.mean(q1_eps_pred_lora_dis), 'q2_eps_pred_lora_dis': np.mean(q2_eps_pred_lora_dis), 'eps_pred_dis': np.mean(eps_pred_dis)}

def evaluate2(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True, eval_maxq_weight: float = 0.07, eval_minqc_weight: float = 0.07
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = agent.eval_actions(obs, train_lora=train_lora, eval_maxq_weight=eval_maxq_weight, eval_minqc_weight=eval_minqc_weight)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs), 'q1_eps_pred_lora_dis': np.mean(q1_eps_pred_lora_dis), 'q2_eps_pred_lora_dis': np.mean(q2_eps_pred_lora_dis), 'eps_pred_dis': np.mean(eps_pred_dis)}

def evaluate_bc(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True
) -> Dict[str, float]:
    # env.config['use_render'] = True
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_bc(obs, train_lora=train_lora)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_ws(
    agent, agent_reward, agent_cost, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(agent_reward, agent_cost, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_composition(
    agent, agent_lora, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    w0 = []
    w2 = []
    # config = {'use_render':True}

    # env.config['use_render'] = True
    
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render(mode="top_down", screen_record=True, screen_size=(500, 500))
                time.sleep(1e-3)
            action, agent = agent.eval_actions_composition(agent_lora, obs)
            # obs1 = np.expand_dims(obs, axis=0)
            # alpha_as = agent.value.apply_fn({'params': agent.value.params}, obs1)
            # ws0 = alpha_as[:,0][:,None].mean()
            # ws1 = alpha_as[:,1][:,None].mean()
            # print("ws0", ws0)
            # print("ws1", ws1)
            # w0.append(alpha_as[:,0][:,None].mean())
            # w2.append(alpha_as[:,1][:,None].mean()/16)
            obs, reward, terminated, truncated, info = env.step(action)
            # video_recorder.record(env)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)
        # video_recorder.save(f"ws_trend.mp4")
    # creat_plot(w0_all=w0, w2_all=w2, save_dir='plot')
    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_score_com(
    agent, agent_lora, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_lora_score_com(agent_lora, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_action_com(
    agent, agent_lora, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_lora_action_com(agent_lora, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_ws3(
    agent, agent_bc, agent_reward, agent_cost, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(agent_bc, agent_reward, agent_cost, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

def evaluate_lora_reward(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True
) -> Dict[str, float]:
    
    if save_video:
        env = RecordVideo(env, './video', episode_trigger=lambda e: True)
    # env = RecordVideo(env, './video', episode_trigger=lambda e: True)
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_lora_reward(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"q_return": np.mean(episode_rets), "q_episode_len": np.mean(episode_lens), "q_cost": np.mean(episode_costs)}

def evaluate_lora_cost(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True
) -> Dict[str, float]:
    if save_video:
        env = RecordVideo(env, './video', episode_trigger=lambda e: True)
    # env = RecordVideo(env, './video', episode_trigger=lambda e: True)
    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions_lora_cost(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"qc_return": np.mean(episode_rets), "qc_episode_len": np.mean(episode_lens), "qc_cost": np.mean(episode_costs)}



def evaluate_pr(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs), "no_safe": np.mean(episode_no_safes)}

def evaluate_demo(
    agent, actions: jnp.ndarray, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True, eval_maxq_weight: float = 0.07, eval_minqc_weight: float = 0.07, 
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes, sums_reward_energy = [], [], [], [], []
    episode_xs, episode_ys = [], []
    episode_ret, episode_energy, episode_len, sum_reward_energy = 0.0, 0.0, 0, 0.0
    episode_x, episode_y = 0.0, 0.0
    actions = actions[0]
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
    
        action, agent, q1_eps_pred_lora_dis, q2_eps_pred_lora_dis, eps_pred_dis = agent.eval_actions(actions, train_lora=train_lora, eval_maxq_weight=eval_maxq_weight, eval_minqc_weight=eval_minqc_weight)

        x = action[0]
        y = action[1]
        energy = get_energy(action)
        reward = get_reward(action)
        sum_reward_energy = reward + energy

        episode_x += x
        episode_y += y
        episode_ret += reward
        episode_energy += energy

        # episode_xs.append(episode_x)
        # episode_ys.append(episode_y)
        # episode_rets.append(episode_ret)
        # episode_energy.append(episode_energy)
        # sums_reward_energy.append(sum_reward_energy)

    return {"return": np.mean(episode_ret/num_episodes), "energy": np.mean(episode_energy/num_episodes), 'sum_reward_energy': np.mean(sum_reward_energy/num_episodes),'x': np.mean(episode_x/num_episodes), 'y': np.mean(episode_y/num_episodes),'q1_eps_pred_lora_dis': np.mean(q1_eps_pred_lora_dis), 'q2_eps_pred_lora_dis': np.mean(q2_eps_pred_lora_dis), 'eps_pred_dis': np.mean(eps_pred_dis)}

def evaluate_bc_demo(
    agent, actions: jnp.ndarray, num_episodes: int, save_video: bool = False, render: bool = False, train_lora: bool = True
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    episode_xs, episode_ys = [], []
    episode_ret, episode_energy, episode_len, sum_reward_energy= 0.0, 0.0, 0, 0.0
    episode_x, episode_y = 0.0, 0.0
    actions = actions[0]
    for _ in trange(num_episodes, desc="Evaluating", leave=False):

        action, agent = agent.eval_actions_bc(actions, train_lora=train_lora)

        x = action[0]
        y = action[1]
        energy = get_energy(action)
        reward = get_reward(action)
        sum_reward_energy = reward + energy
        episode_x += x
        episode_y += y
        episode_energy += energy
        episode_ret += reward

        # episode_xs.append(episode_x)
        # episode_ys.append(episode_y)
        # episode_rets.append(episode_ret)
        # episode_energy.append(episode_energy)

    return {"return": np.mean(episode_ret/num_episodes), "energy": np.mean(episode_energy/num_episodes),'sum_reward_energy': np.mean(sum_reward_energy/num_episodes), 'x': np.mean(episode_x/num_episodes), 'y': np.mean(episode_y/num_episodes)}

@jax.jit
def get_energy_jax(points):
    # Ensure points is a JAX array
    points = jnp.array(points)

    # Define center coordinates
    sqrt2 = jnp.sqrt(2)
    centers = jnp.array([
        (0, 1),
        (-1. / sqrt2, 1. / sqrt2),
        (-1, 0),
        (-1. / sqrt2, -1. / sqrt2),
        (0, -1),
        (1. / sqrt2, -1. / sqrt2),
        (1, 0),
        (1. / sqrt2, 1. / sqrt2),
    ])
    scale = 1
    # Scale the centers
    scaled_centers = scale * centers
    radius = 0.5

    # Compute the distances from each point to each center
    distances = jnp.sqrt(((points[:, None, :] - scaled_centers[None, :, :]) ** 2).sum(axis=-1))

    # Find the index of the closest center within the radius
    within_radius = (distances <= radius)
    energy = jnp.where(within_radius, jnp.arange(scaled_centers.shape[0])[None, :], jnp.inf)

    # Choose the minimum index or default to a special value if no center is close
    energy = jnp.min(energy, axis=1)
    default_value = jnp.full(energy.shape, 7)  # Default index when no center is within radius
    # replace energy 0 with 7
    energy = jnp.where(energy==0, default_value, energy)

    return energy

@jax.jit
def get_reward_jax(points):

    points = jnp.array(points)

    centers = jnp.array([
        (0, 1), 
        (-1. / jnp.sqrt(2), 1. / jnp.sqrt(2)),
        (-1, 0), 
        (-1. / jnp.sqrt(2), -1. / jnp.sqrt(2)),
        (0, -1),
        (1. / jnp.sqrt(2), -1. / jnp.sqrt(2)),
        (1, 0), 
        (1. / jnp.sqrt(2), 1. / jnp.sqrt(2)),
    ])
    scale = 1

    scaled_centers = scale * centers
    radius = 0.5

    distances = jnp.sqrt(jnp.sum((points[:, None, :] - scaled_centers[None, :, :]) ** 2, axis=2))
    within_radius = distances <= radius

    center_rewards = 6 - jnp.arange(len(centers))
    rewards = jnp.where(within_radius, center_rewards, 0)
    rewards = jnp.max(rewards, axis=1)

    special_center = jnp.array([0, scale])
    distance_to_special = jnp.sqrt(jnp.sum((points - special_center) ** 2, axis=1))
    special_rewards = jnp.where(distance_to_special <= radius, 6, 0)

    final_rewards = jnp.maximum(rewards, special_rewards)

    return final_rewards


def get_energy(points):
    # example the points is [[],]
    # assert isinstance(points, list)

    if not isinstance(points[0], list):
        points = [points]

    # assert isinstance(points[0], list)

    # 定义中心点的坐标
    centers = jnp.array([
        (0, 1), 
        (-1. / jnp.sqrt(2), 1. / jnp.sqrt(2)),
        (-1, 0), 
        (-1. / jnp.sqrt(2), -1. / jnp.sqrt(2)),
        (0, -1),
        (1. / jnp.sqrt(2), -1. / jnp.sqrt(2)),
        (1, 0), 
        (1. / jnp.sqrt(2), 1. / jnp.sqrt(2)),
    ])
    scale = 1

    scaled_centers = [(scale * x, scale * y) for x, y in centers]
    radius = 0.5  
    energy = [0] * len(points) 
    special_center = (0, scale) 
    for idx, point in enumerate(points):
        for center_idx, center in enumerate(scaled_centers):
            distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            if distance <= radius:
                energy[idx] = center_idx
            # energy[idx] = jnp.where(distance <= radius, center_idx, energy[idx])
        distance_to_special = np.sqrt((point[0] - special_center[0])**2 + (point[1] - special_center[1])**2)
        if distance_to_special <= radius:
            energy[idx] = np.array(7)
        # energy[idx] = jnp.where(distance_to_special <= radius, 7, energy[idx])

    return np.mean(energy)
    # return energy

def get_reward(points):

    # example the points is [[],]
    # assert isinstance(points, list)

    if not isinstance(points[0], list):
        points = [points]

    # assert isinstance(points[0], list)


    centers = [
        (0, 1), 
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1, 0), 
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
        (0, -1),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (1, 0), 
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
    ]
    scale = 1

    scaled_centers = [(scale * x, scale * y) for x, y in centers]
    radius = 0.5  
    reward = [0] * len(points)  
    special_center = (0, scale)  
    for idx, point in enumerate(points):

        for center_idx, center in enumerate(scaled_centers):
            distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
            if distance <= radius:
                reward[idx] = 6 - center_idx
                if reward[idx] < 1:
                    reward[idx] = 7
        distance_to_special = np.sqrt((point[0] - special_center[0])**2 + (point[1] - special_center[1])**2)
        if distance_to_special <= radius:
            reward[idx] = 6
            
    return np.mean(reward)
