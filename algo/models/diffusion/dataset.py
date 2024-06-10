import os
import pickle

import numpy as np
import torch


def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
):
    indices = list()

    for i in range(len(episode_ends)):

        start_idx = 0
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [i, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
        representation_type,
        traj_list,
        sequence_length,
        file_idx,
        buffer_start_idx,
        buffer_end_idx,
        sample_start_idx,
        sample_end_idx,
):
    result = dict()
    tactile_folder = traj_list[file_idx][:-7].replace('obs', 'tactile')
    img_folder = traj_list[file_idx][:-7].replace('obs', 'img')

    if 'tactile' in representation_type:
        tactile = np.stack([np.load(os.path.join(tactile_folder, f'tactile_{i}.npz'))['tactile'] for i in
                            range(buffer_start_idx, buffer_end_idx)])

        # Handle filling for tactile data
        tactile_filled = np.zeros((sequence_length,) + tactile.shape[1:], dtype=tactile.dtype)
        if sample_start_idx > 0:
            tactile_filled[:sample_start_idx] = tactile[0]
        if sample_end_idx < sequence_length:
            tactile_filled[sample_end_idx:] = tactile[-1]
        tactile_filled[sample_start_idx:sample_end_idx] = tactile
        result['tactile'] = tactile_filled

    if 'img' in representation_type:
        img = np.stack(
            [np.load(os.path.join(img_folder, f'img_{i}.npz'))['img'] for i in range(buffer_start_idx, buffer_end_idx)])

        # Handle filling for image data
        img_filled = np.zeros((sequence_length,) + img.shape[1:], dtype=img.dtype)
        if sample_start_idx > 0:
            img_filled[:sample_start_idx] = img[0]
        if sample_end_idx < sequence_length:
            img_filled[sample_end_idx:] = img[-1]
        img_filled[sample_start_idx:sample_end_idx] = img
        result['img'] = img_filled

    # Load the training data
    train_data = np.load(traj_list[file_idx])

    for key, input_arr in train_data.items():
        if key not in representation_type:
            continue

        sample = input_arr[buffer_start_idx:buffer_end_idx]

        data = sample

        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data

    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    if np.any(stats["max"] > 1e5) or np.any(stats["min"] < -1e5):
        raise ValueError("data out of range")
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]
    return data


from pathlib import Path
from tqdm import tqdm
import random


class DataNormalizer:
    def __init__(self, cfg, file_list):
        self.cfg = cfg
        self.normalize_keys = self.cfg.normalize_keys
        self.normalization_path = self.cfg.normalize_file
        self.normalize_dict = {"mean": {}, "std": {}}
        self.file_list = file_list
        self.remove_failed_trajectories()
        self.run()

    def ensure_directory_exists(self, path):
        """Ensure the directory for the given path exists."""
        directory = Path(path).parent.absolute()
        directory.mkdir(parents=True, exist_ok=True)

    def remove_failed_trajectories(self):
        """Remove files corresponding to failed trajectories."""
        print('Removing failed trajectories')
        cleaned_file_list = []
        for file in tqdm(self.file_list, desc="Cleaning files"):
            try:
                d = np.load(file)
                done_idx = d['done'].nonzero()[0]
                if len(done_idx) > 0:  # Ensure done_idx is not empty
                    cleaned_file_list.append(file)
                else:
                    os.remove(file)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                print(f"Error processing {file}: {e}")
                os.remove(file)
        self.file_list = cleaned_file_list

    def load_or_create_normalization_file(self):
        """Load the normalization file if it exists, otherwise create it."""
        if os.path.exists(self.normalization_path):
            with open(self.normalization_path, 'rb') as f:
                self.normalize_dict = pickle.load(f)
        else:
            self.create_normalization_file()

    def create_normalization_file(self):
        """Create a new normalization file."""
        for norm_key in self.normalize_keys:
            print(f'Creating new normalization file for {norm_key}')
            data = self.aggregate_data(norm_key)
            self.calculate_normalization_values(data, norm_key)
        self.save_normalization_file()

    def aggregate_data(self, norm_key):
        """Aggregate data for the given normalization key."""
        data = []
        file_list = self.file_list
        for file in tqdm(random.sample(file_list, len(file_list)), desc=f"Processing {norm_key}"):
            try:
                d = np.load(file)
                done_idx = d['done'].nonzero()[0][-1]
                data.append(d[norm_key][:done_idx, :])
            except Exception as e:
                print(f"{file} could not be processed: {e}")
        return np.concatenate(data, axis=0)

    def calculate_normalization_values(self, data, norm_key):
        """Calculate mean and standard deviation for the given data."""
        self.normalize_dict['mean'][norm_key] = np.mean(data, axis=0)
        self.normalize_dict['std'][norm_key] = np.std(data, axis=0)

    def save_normalization_file(self):
        """Save the normalization values to file."""
        print(f'Saved new normalization file at: {self.normalization_path}')
        with open(self.normalization_path, 'wb') as f:
            pickle.dump(self.normalize_dict, f)

    def run(self):
        """Main method to run the process."""
        self.ensure_directory_exists(self.normalization_path)
        self.load_or_create_normalization_file()


class MemmapLoader:
    def __init__(self, path):
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta_data = pickle.load(f)

        print("Meta Data:", meta_data)
        self.fps = {}

        self.length = None
        for key, (shape, dtype) in meta_data.items():
            self.fps[key] = np.memmap(
                os.path.join(path, key + ".dat"), dtype=dtype, shape=shape, mode="r"
            )
            if self.length is None:
                self.length = shape[0]
            else:
                assert self.length == shape[0]

    def __getitem__(self, index):
        rets = {}
        for key in self.fps.keys():
            value = self.fps[key]
            value = value[index]
            value_cp = np.empty(dtype=value.dtype, shape=value.shape)
            value_cp[:] = value
            rets[key] = value_cp
        return rets

    def __length__(self):
        return self.length


# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            traj_list: list,
            representation_type: list,
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int,
            stats: dict = None,
            img_transform=None,
            tactile_transform=None,
            get_img=None,
            load_img: bool = False,
            binarize_tactile: bool = False,
            state_noise: float = 0.0,
            img_dim: tuple = (180, 320),
            tactile_dim: tuple = (0, 0),
    ):

        self.state_noise = state_noise
        self.img_dim = img_dim
        self.tactile_dim = tactile_dim
        self.count = 0
        # self.memmap_loader = None
        # if "memmap_loader_path" in data.keys():
        #     self.memmap_loader = MemmapLoader(data["memmap_loader_path"])

        self.representation_type = representation_type
        self.img_transform = img_transform
        self.tactile_transform = tactile_transform

        self.get_img = get_img
        self.load_img = load_img

        episode_ends = []

        for file_idx, file in enumerate(traj_list):
            data = np.load(file)
            done = data["done"]
            data_length = done.nonzero()[0][-1]
            episode_ends.append(data_length)

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.traj_list = traj_list
        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.binarize_tactile = binarize_tactile

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            file_idx,
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            representation_type=self.representation_type + ['action'],
            traj_list=self.traj_list,
            sequence_length=self.pred_horizon,
            file_idx=file_idx,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        for k in self.representation_type:
            # discard unused observations
            nsample[k] = nsample[k][: self.obs_horizon]

            nsample[k] = torch.tensor(nsample[k], dtype=torch.float32)

            if self.state_noise > 0.0:
                # add noise to the state
                nsample[k] = nsample[k] + torch.randn_like(nsample[k]) * self.state_noise

        nsample["action"] = torch.tensor(nsample["action"], dtype=torch.float32)

        return nsample

class Dataset2(torch.utils.data.Dataset):
    def __init__(
            self,
            traj_list: list,
            representation_type: list,
            pred_horizon: int,
            obs_horizon: int,
            action_horizon: int,
            stats: dict = None,
            img_transform=None,
            tactile_transform=None,
            binarize_tactile: bool = False,
            state_noise: float = 0.0,
    ):

        self.state_noise = state_noise
        self.count = 0
        # self.memmap_loader = None
        # if "memmap_loader_path" in data.keys():
        #     self.memmap_loader = MemmapLoader(data["memmap_loader_path"])
        self.to_torch = lambda x: torch.from_numpy(x).float()

        self.representation_type = representation_type
        self.img_transform = img_transform
        self.tactile_transform = tactile_transform
        self.sequence_length = pred_horizon
        self.stride = pred_horizon

        self.indices_per_trajectory = []
        for file_idx, file in enumerate(traj_list):
            data = np.load(file)
            done = data["done"]
            done_idx = done.nonzero()[0][-1]
            total_len = done_idx

            if total_len >= self.sequence_length:
                num_subsequences = (total_len - self.sequence_length) // self.stride + 1
                self.indices_per_trajectory.extend([(file_idx, i * self.stride) for i in range(num_subsequences)])
        print('Total sub trajectories:', len(self.indices_per_trajectory))


        self.traj_list = traj_list
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.binarize_tactile = binarize_tactile

    def __len__(self):
        return int((len(self.indices_per_trajectory)))

    def extract_sequence(self, data, key, start_idx):
        # Extract a sequence of specific length from the array
        return data[key][start_idx:start_idx + self.sequence_length]

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        file_idx, start_idx = self.indices_per_trajectory[idx]
        file_path = self.traj_list[file_idx]
        data = np.load(file_path)

        nsample = {key: self.extract_sequence(data, key, start_idx) for key in self.representation_type}

        for k in self.representation_type:
            # discard unused observations
            nsample[k] = nsample[k][: self.obs_horizon]
            nsample[k] = torch.tensor(nsample[k], dtype=torch.float32)

            if k == 'img':
                nsample[k] = self.img_transform(nsample[k])
            elif k == 'tactile':
                nsample[k] = self.tactile_transform(nsample[k])
            elif self.state_noise > 0.0:
                # add noise to the state
                nsample[k] = nsample[k] + torch.randn_like(nsample[k]) * self.state_noise

        action = self.extract_sequence(data, "action", start_idx)
        nsample["action"] = torch.tensor(action, dtype=torch.float32)

        return nsample
