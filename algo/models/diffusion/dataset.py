import os
import pickle

import numpy as np
import torch
from typing import List, Dict, Any
from algo.models.diffusion.utils import convert_trajectory

def create_sample_indices(
        episode_ends: List,
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

        # range stops one idx before end; jump by 1 step - there is overlap
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


def fill_sequence(data: np.ndarray,
                  grasp_state: np.ndarray,
                  sequence_length: int,
                  sample_start_idx: int,
                  sample_end_idx: int,
                  cond_on_grasp: bool = False,
                  is_action: bool = False) -> np.ndarray:

    if cond_on_grasp and not is_action:
        sequence_length += 1
        filled_data = np.zeros((sequence_length,) + data.shape[1:], dtype=data.dtype)

        # Fill the initial portion with the first element, shifted by one index
        filled_data[1:sample_start_idx + 1] = data[0]

        # Fill the end portion with the last element, if needed, shifted by one index
        if sample_end_idx < sequence_length - 1:
            filled_data[sample_end_idx + 1:] = data[-1]

        # Fill the main portion of the data, shifted by one index
        filled_data[sample_start_idx + 1:sample_end_idx + 1] = data

        # Add the first element of the trajectory at the very start of the sequence
        filled_data[0] = grasp_state

    else:
        filled_data = np.zeros((sequence_length,) + data.shape[1:], dtype=data.dtype)

        # Fill the initial portion if the sample_start_idx is greater than 0
        if sample_start_idx > 0:
            filled_data[:sample_start_idx] = data[0]

        # Fill the end portion if sample_end_idx is less than sequence_length
        if sample_end_idx < sequence_length:
            filled_data[sample_end_idx:] = data[-1]

        # Fill the main portion of the data
        filled_data[sample_start_idx:sample_end_idx] = data

    return filled_data


def load_grasp_state(folder: str, prefix: str) -> np.ndarray:
    return np.load(os.path.join(folder, f'{prefix}_1.npz'))[prefix][0]


def load_data(folder: str, prefix: str, buffer_start_idx: int, buffer_end_idx: int) -> np.ndarray:
    return np.stack([np.load(os.path.join(folder, f'{prefix}_{i}.npz'))[prefix] for i in range(buffer_start_idx,
                                                                                               buffer_end_idx)])


def sample_sequence(
        representation_type: List[str],
        traj_list: List[str],
        sequence_length: int,
        file_idx: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int,
        cond_on_grasp: bool = False,
        ) -> Dict[str, Any]:

    result = {}

    tactile_folder = traj_list[file_idx][:-7].replace('obs', 'tactile')
    img_folder = traj_list[file_idx][:-7].replace('obs', 'img')
    train_data = np.load(traj_list[file_idx])

    grasp_state = {}
    if cond_on_grasp:
        if 'tactile' in representation_type:
            grasp_state['tactile'] = load_grasp_state(tactile_folder, 'tactile')
        if 'img' in representation_type:
            grasp_state['img'] = load_grasp_state(img_folder, 'img')

        for key, input_arr in train_data.items():
            if key in representation_type:
                grasp_state[key] = input_arr[0]

    if 'tactile' in representation_type:
        tactile = load_data(tactile_folder, 'tactile', buffer_start_idx, buffer_end_idx)
        result['tactile'] = fill_sequence(tactile, grasp_state.get('tactile'), sequence_length, sample_start_idx,
                                          sample_end_idx, cond_on_grasp)

    if 'img' in representation_type:
        img = load_data(img_folder, 'img', buffer_start_idx, buffer_end_idx)
        result['img'] = fill_sequence(img, grasp_state.get('img'), sequence_length, sample_start_idx, sample_end_idx,
                                      cond_on_grasp)

    for key, input_arr in train_data.items():
        if key not in representation_type:
            continue

        sample = input_arr[buffer_start_idx:buffer_end_idx]
        grasp_state_value = grasp_state.get(key) if cond_on_grasp else None
        result[key] = fill_sequence(sample, grasp_state_value, sequence_length, sample_start_idx, sample_end_idx,
                                    cond_on_grasp, key == 'action')

        if key == 'eef_pos':
            result[key] = convert_trajectory(result[key])

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
import pytorch3d.transforms as pt
import functools
from typing import Union
from scipy.spatial.transform import Rotation

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
                 from_rep='quaternion',
                 to_rep='rotation_6d',
                 from_convention=None,
                 to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)

class DataNormalizer:
    def __init__(self, cfg, file_list):
        self.cfg = cfg
        self.normalize_keys = self.cfg.normalize_keys
        self.normalization_path = self.cfg.normalize_file
        self.stats = {"mean": {}, "std": {}}
        self.file_list = file_list
        self.remove_failed_trajectories()
        self.rot_tf = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')
        self.rot_tf_from_quat = RotationTransformer()
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
                self.stats = pickle.load(f)
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
        if norm_key == 'eef_pos':
            eef_pos_rot6d = np.concatenate((data[:, :3],
                                            self.rot_tf.forward(data[:, 3:].reshape(data.shape[0], 3, 3))), axis=1)
            self.stats['mean']['eef_pos'] = np.mean(eef_pos_rot6d, axis=0)
            self.stats['std']['eef_pos'] = np.std(eef_pos_rot6d, axis=0)
            # self.stats['mean'][norm_key] = np.mean(data, axis=0)
            # self.stats['std'][norm_key] = np.std(data, axis=0)
        else:
            # Handle other normalization obs_keys
            self.stats['mean'][norm_key] = np.mean(data, axis=0)
            self.stats['std'][norm_key] = np.std(data, axis=0)

    def save_normalization_file(self):
        """Save the normalization values to file."""
        print(f'Saved new normalization file at: {self.normalization_path}')
        with open(self.normalization_path, 'wb') as f:
            pickle.dump(self.stats, f)

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
            state_noise: float = 0.0,
            img_dim: tuple = (180, 320),
            tactile_dim: tuple = (0, 0),
            cond_on_grasp: bool = False,
    ):

        self.state_noise = state_noise
        self.img_dim = img_dim
        self.tactile_dim = tactile_dim
        self.count = 0

        self.representation_type = representation_type
        self.img_transform = img_transform
        self.tactile_transform = tactile_transform
        self.cond_on_grasp = cond_on_grasp
        self.get_img = get_img
        self.rot_tf = RotationTransformer(from_rep='matrix', to_rep='rotation_6d')

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
            cond_on_grasp=self.cond_on_grasp
        )

        for k in self.representation_type:
            # discard unused observations
            nsample[k] = nsample[k][: self.obs_horizon]

            nsample[k] = torch.tensor(nsample[k], dtype=torch.float32)

            if k == 'eef_pos':
                eef_pos = nsample[k]
                eef_pos_rot6d = np.concatenate((eef_pos[:, :3],
                                                self.rot_tf.forward(eef_pos[:, 3:].reshape(eef_pos.shape[0], 3, 3))), axis=1)
                nsample[k] = eef_pos_rot6d

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
            state_noise: float = 0.0,
    ):

        self.state_noise = state_noise
        self.count = 0
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
