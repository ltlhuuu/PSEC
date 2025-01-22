import os
import gym
import dsrl
import numpy as np
from jaxrl5.data.dataset_dsrl import Dataset
import h5py

def inf_train_gen(data, batch_size=200):
    print(data)
    if data == "swissroll":
        print(data)
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.sum(data**2, axis=-1,keepdims=True) / 9.0
    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data
    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        center_dist = X[:,0]**2 + X[:,1]**2
        energy = np.zeros_like(center_dist)

        energy[(center_dist >=8.5)] = 0.667 
        energy[(center_dist >=5.0) & (center_dist <8.5)] = 0.333 
        energy[(center_dist >=2.0) & (center_dist <5.0)] = 1.0 
        energy[(center_dist <2.0)] = 0.0

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), energy[:,None]

    elif data == "moons":
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data.astype(np.float32), (y > 0.5).astype(np.float32)[:,None]

    # elif data == "8gaussians":
    #     scale = 4.
    #     centers = [
    #                (0, 1), 
    #                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
    #                (-1, 0), 
    #                (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    #                (0, -1),
    #                (1. / np.sqrt(2), -1. / np.sqrt(2)),
    #                 (1, 0), 
    #                (1. / np.sqrt(2), 1. / np.sqrt(2)),
    #                ]
        
        
    #     centers = [(scale * x, scale * y) for x, y in centers]

    #     dataset = []
    #     indexes = []
    #     for i in range(batch_size):
    #         point = np.random.randn(2) * 0.5
    #         idx = np.random.randint(8)
    #         center = centers[idx]
    #         point[0] += center[0]
    #         point[1] += center[1]
    #         indexes.append(idx)
    #         dataset.append(point)
    #     dataset = np.array(dataset, dtype="float32")
    #     dataset /= 1.414
    #     return dataset, np.array(indexes, dtype="float32")[:,None] / 7.0

    elif data == "8gaussians-multitarget-rightup":
        scale = 1.
        # 
        centers_left = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        
        
        centers_left = [(scale * x, scale * y) for x, y in centers_left]
        # centers_right = [(scale * x, scale * y) for x, y in centers_right]

        dataset = []
        indexes_left = []
        indexes_right = []
        for i in range(batch_size):
            # left centers
            # point = np.random.randn(2) * 0.1
            point = np.random.randn(2) * 0.08
            idx_left = np.random.randint(8)
            idx_right = 6 - idx_left
            center_left = centers_left[idx_left]
            point[0] += center_left[0]
            point[1] += center_left[1]
            if idx_left == 7:
                idx_right = 7
            if idx_left == 6:
                idx_right = 7
            if idx_left == 0:
                idx_left = 7
                idx_right = 6
            indexes_left.append(idx_left)
            indexes_right.append(idx_right)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset, np.array(indexes_left, dtype="float32")[:,None], np.array(indexes_right, dtype="float32")[:,None]
    
    elif data == "8gaussians-multitarget":
        scale = 1.
        # 
        centers_left = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        
        
        centers_left = [(scale * x, scale * y) for x, y in centers_left]
        # centers_right = [(scale * x, scale * y) for x, y in centers_right]

        dataset = []
        indexes_left = []
        indexes_right = []
        for i in range(batch_size):
            # left centers
            # point = np.random.randn(2) * 0.1
            point = np.random.randn(2) * 0.08
            idx_left = np.random.randint(8)
            idx_right = 8 - idx_left
            center_left = centers_left[idx_left]
            point[0] += center_left[0]
            point[1] += center_left[1]
            # if idx_left == 7:
            #     idx_right = 7
            # if idx_left == 6:
            #     idx_right = 7
            if idx_left == 0:
                idx_left = 7
                idx_right = 7
            indexes_left.append(idx_left)
            indexes_right.append(idx_right)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset, np.array(indexes_left, dtype="float32")[:,None], np.array(indexes_right, dtype="float32")[:,None]
    
    elif data == "8gaussians-multitarget-sparse":
        scale = 1.
        # 
        centers_left = [
                   (0, 1), 
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1, 0), 
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (0, -1),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (1, 0), 
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   ]
        
        
        centers_left = [(scale * x, scale * y) for x, y in centers_left]
        # centers_right = [(scale * x, scale * y) for x, y in centers_right]

        dataset = []
        indexes_left = []
        indexes_right = []
        for i in range(batch_size):
            # left centers
            point = np.random.randn(2) * 0.08
            idx_left = np.random.randint(8)
            idx_right = 8 - idx_left
            center_left = centers_left[idx_left]
            point[0] += center_left[0]
            point[1] += center_left[1]
            if idx_left == 0:
                idx_left = 7
                idx_right = 7
            if idx_left == 0 or idx_left == 1 or idx_left == 7:
                indexes_left.append(idx_left)
                indexes_right.append(idx_right)
            else:
                indexes_left.append(0)
                indexes_right.append(0)
            
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        # dataset /= 1.414
        return dataset, np.array(indexes_left, dtype="float32")[:,None], np.array(indexes_right, dtype="float32")[:,None]

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x, np.clip((1-np.concatenate([n,n]) / 10),0,1)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        points = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        points_x = points[:,0]
        judger = ((points_x > 0) & (points_x <= 2)) | ((points_x <= -2))
        return points, judger.astype(np.float32)[:,None]

    elif data == "line":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        assert False


class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, critic_type="qc", data_location=None, cost_scale=1., ratio = 1.0):

        if data_location is not None:
            # imitation data
            dataset_dict = env.get_dataset(h5path=data_location)
            print('=========Data loading=========')
            print('Load imitation data from:', data_location)
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost,
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"], dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict["timeouts"]

        else:
            # DSRL
            if ratio == 1.0:
                dataset_dict = env.get_dataset()
            else:
                _, dataset_name = os.path.split(env.dataset_url)
                file_list = dataset_name.split('-')
                ratio_num = int(float(file_list[-1].split('.')[0]) * ratio)
                dataset_ratio = '-'.join(file_list[:-1]) + '-' + str(ratio_num) + '-' + str(ratio) + '.hdf5'
                dataset_dict = env.get_dataset(os.path.join('data', dataset_ratio))
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost, 
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            print('data_num', dataset_dict['actions'].shape[0])
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                                dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict['timeouts']

            if critic_type == "hj":
                dataset_dict['costs'] = np.where(dataset_dict['costs']>0, 1*cost_scale, -1)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        super().__init__(dataset_dict)

class Toy_dataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, datanum=1000000):
        # assert name in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
        dataset_dict = {}
        print('=========Data loading=========')
        print('Load point robot data from:', env)
        self.datanum =datanum
        self.env = env
        self.datas, self.energy, self.reward = inf_train_gen(self.env, batch_size=2048)
        dataset_dict["actions"] = np.array(self.datas)
        dataset_dict["energy"] = np.array(self.energy)
        dataset_dict["reward"] = np.array(self.reward)

        self.datadim = 2
        # if clip_to_eps:
        #     lim = 1 - eps
        #     dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        # dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        # del dataset_dict['dones']

        super().__init__(dataset_dict)
      
    # def __getitem__(self, index):
    #     return {"a": self.datas[index], "e": self.energy[index], "r": self.reward[index]}, 

    # def __add__(self, other):
    #     raise NotImplementedError

    # def __len__(self):
    #     return self.datanum
