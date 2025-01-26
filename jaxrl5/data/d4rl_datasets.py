import d4rl
import gym
import numpy as np
import mjrl
from jaxrl5.data.dataset import Dataset
import pickle as pkl

class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, seed: int=42, clip_to_eps: bool = True, eps: float = 1e-5):
        try:
            dataset_dict = d4rl.qlearning_dataset(env)
        except:
            dataset_dict = mjrl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict, seed)
class DynamicDataset(Dataset):
    def __init__(self, dataset_path, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        with open(dataset_path, 'rb') as f:
            dataset = pkl.load(f)
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []
        for data in dataset:
            for i in range(len(data['observations'])-1):
                obs_.append(data['observations'][i])
                action_.append(data['actions'][i])
                next_obs_.append(data['observations'][i+1])
                reward_.append(data['rewards'][i])
                # done_.append(data['terminals'][i])
            
        dataset_dict = {
            "observations": np.array(obs_).reshape(-1, 17),
            "actions": np.array(action_).reshape(-1, 6),
            "next_observations": np.array(next_obs_).reshape(-1, 17),
            "rewards": np.array(reward_).reshape(-1),
            # "terminals": np.array(done_).reshape(-1),
        }
        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                # or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        # dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        dataset_dict["masks"] = 1.0 - dones
        # del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)

class ToyCarDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):

        dataset = env.get_dataset()
        dataset_dict = {
            "observations": np.array(dataset["observations"]),
            "next_observations": np.array(dataset["next_observations"]),
            "actions": np.array(dataset["actions"]),
            "costs": np.array(dataset["costs"]),
            "dones": np.array(dataset["terminals"]),
            "distances": np.array(dataset["distances"]),
            "masks": 1.0 - np.array(dataset["terminals"]),
        }
        print("Loaded dataset from environment")
        # print("env_max_steps", env.max_episode_steps)
        # print("mean_episode_cost", env._max_episode_steps * np.mean(np.array(dataset["costs"])))
        # print("mean_episode_distance", env._max_episode_steps * np.mean(np.array(dataset["distances"])))


        super().__init__(dataset_dict)
