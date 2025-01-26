from typing import Dict

import gym
import numpy as np

from jaxrl5.wrappers.wandb_video import WANDBVideo
from tqdm.auto import trange

def pad_sequences(sequences, pad_value=0.0):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.full((len(sequences), max_len), pad_value)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences
def evaluate_ws(
    agent, agent_bc, agent_lora, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(agent_bc, agent_lora, observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}

def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}

def evaluate_lora(
    agent, env: gym.Env, seed: int, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    env.seed(seed=seed)

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
    # for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions_lora(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}

def evaluate_composition(
    agent, agent_lora, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions_composition(agent_lora, observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}


def evaluate_toy(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    episode_costs, episode_lens, obstacle_position = [], [], []
    action_all = []
    distance_all = []
    states_all = []
    for _ in range(num_episodes):
        episode_cost, episode_len= 0.0, 0
        state_list = []
        action_list = []
        distance_list = []
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(observation)
            # observation, costs, done, _ = env.step(action)
            observation, cost, done, info = env.step(action)
            distance = info["distance"]
            state_list.append(observation)
            distance_list.append(distance)
            episode_cost += cost
            episode_len += 1
            action_list.append(action)
        states_all.append(state_list)
        distance_all.append(distance_list)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)
        obstacle_position.append(observation[1])
        action_all.append(action_list)
    # Pad sequences to ensure homogeneous shapes
    states_all = pad_sequences(states_all)
    distance_all = pad_sequences(distance_all)
    action_all = pad_sequences(action_all)
    np.save("dataset/states_all.npy", states_all)
    np.save("dataset/distance_all.npy", distance_all)
    np.save("dataset/action_all.npy", action_all)
    return { "cost": np.mean(episode_costs), "len": np.mean(episode_lens), "obstacle_position": np.mean(obstacle_position)}

def implicit_evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.sample_implicit_policy(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}
