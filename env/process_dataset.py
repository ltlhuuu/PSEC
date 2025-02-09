from collections import defaultdict
import os
current_path = os.getcwd()
import sys
sys.path.insert(0, current_path)
import os.path as osp
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from env.utils import get_trajectory_info, filter_trajectory
from env.env_name import env_name
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--env_id', type=int, default=0)
    parser.add_argument('--output', '-o', type=str, default='car-circle')
    parser.add_argument('--dir', '-d', type=str, default='env/log')
    parser.add_argument('--cmax', type=int, default=50)
    parser.add_argument('--cmin', type=int, default=0)
    parser.add_argument('--rmax', type=float, default=300)
    parser.add_argument('--rmin', type=float, default=200)
    parser.add_argument('--cbins', type=int, default=60)
    parser.add_argument('--rbins', type=int, default=50)
    parser.add_argument('--npb', type=int, default=1000)
    parser.add_argument('--save', action="store_true")
    args = parser.parse_args()
    dsrl_path = '/home/liutl/.dsrl/datasets/'
    # fing the h5py files have the env name from the given path
    all_files = os.listdir(dsrl_path)
    env_name = env_name[args.env_id]
    filtered_files = [f for f in all_files if env_name in f and f.endswith('.hdf5')]
    args.files = [osp.join(dsrl_path, f) for f in filtered_files]
    args.output = env_name.replace('Safe', 'Offline').replace('MetaDrive', 'Metadrive') + '-v0'
    hfiles = []
    for file in args.files:
        hfiles.append(h5py.File(file, 'r'))
    
    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]
    float_32_keys = ['observations', 'next_observatios', 'actions', 'rewards',' costs']

    print("*" * 10, "concatenating dataset from", "*" * 10)

    for file in args.files:
        print("*" * 10, file, "*" * 10)

    dataset_dict = {}
    for k in keys:
        d = [hfile[k] for hfile in hfiles]
        combined = np.concatenate(d, axis=0)
        # convert the data to the same float 32 data format
        if k in float_32_keys:
            combined = combined.astype(np.float32)
        print(k, combined.shape, combined.dtype)
        dataset_dict[k] = combined
    # process the dataset to the SDT format:
    # traj[i] is the i-th trajectory (a dict)
    rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)
    traj = []
    for i in trange(len(rew_ret), desc="Processing trajectories..."):
        start = start_index[i]
        end = end_index[i] + 1
        one_traj = {k: dataset_dict[k][start:end] for k in keys}
        traj.append(one_traj)
    print(f"Total number of trajectories : {len(traj)}")

    # plot the original dataset cost-reward figure 
    output_name = args.output + '-' + str(int(args.cmax))
    plt.scatter(cost_ret, rew_ret, s=2)
    plt.xlabel("Costs")
    plt.ylabel("Rewards")
    fig_path = 'env/fig'
    output_path = osp.join(fig_path, output_name + "_before_filter.png")
    if not osp.exists("fig"):
        os.makedirs("fig")
    plt.savefig(output_path)
    plt.clf()
    
    # downsampling the trajectories by grid filter
    cost_ret_filter, rew_ret_filter, traj = filter_trajectory(
        cost_ret,
        rew_ret,
        traj,
        cost_min=args.cmin,
        cost_max=args.cmax,
        rew_min=args.rmin,
        rew_max=args.rmax,
        cost_bins=args.cbins,
        rew_bins=args.rbins,
        max_num_per_bin=args.npb
    )

    print(f"Number of trajectories after filtering: {len(traj)}")

    # plot the filtered dataset cost-reward figure
    plt.scatter(cost_ret, rew_ret, s=2)
    plt.scatter(cost_ret_filter, rew_ret_filter, c='r', s=2)
    plt.xlabel("Costs")
    plt.ylabel("Rewards")

    if not osp.exists(fig_path):
        os.makedirs(fig_path)
    output_path = osp.join(fig_path, output_name  + '-' + str(len(traj)) + "_after_filter.png" )
    plt.savefig(output_path)

    # process the trajectory data back to the d4rl data format
    dataset = defaultdict(list)
    for d in traj:
        for k in keys:
            dataset[k].append(d[k])
    for k in keys:
        dataset[k] = np.squeeze(np.concatenate(dataset[k], axis=0))
        print(k, np.array(dataset[k]).shape, dataset[k].dtype)
    
    output_name = args.output + '-' + str(int(args.cmax)) + '-' + str(len(traj))

    # store the data
    if args.save:
        if not osp.exists(args.dir):
            os.makedirs(args.dir)
        output_name = output_name + ".hdf5"
        output_file = osp.join(args.dir, output_name)
        outf = h5py.File(output_file, 'w')
        for k in keys:
            outf.create_dataset(k, data=dataset[k], compression='gzip')
        outf.close()
