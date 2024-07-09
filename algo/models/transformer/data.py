from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as F

import os
import pickle
from scipy.spatial.transform import Rotation
from pathlib import Path
from tqdm import tqdm
import random
import cv2


def get_last_sequence(input_tensor, progress_buf, sequence_length):
    """
    Adjusts the input tensor to the desired sequence length by padding with zeros if the
    actual sequence length is less than the desired, or by selecting the most recent
    entries up to the desired sequence length if the actual sequence is longer.

    Parameters:
    - input_tensor: A tensor with shape [E, T, ...], where E is the number of environments,
      and T is the temporal dimension.
    - progress_buf: A tensor or an array indicating the actual sequence length for each environment.
    - sequence_length: The desired sequence length.

    Returns:
    - A tensor adjusted to [E, sequence_length, ...].
    """

    E, _, *other_dims = input_tensor.shape
    adjusted_tensor = torch.zeros(E, sequence_length, *other_dims, device=input_tensor.device,
                                  dtype=torch.float32) + 1e-6

    for env_id in range(E):
        actual_seq_len = progress_buf[env_id]
        if actual_seq_len < sequence_length:
            # If the sequence is shorter, pad the beginning with zeros
            actual_seq_len = progress_buf[env_id] + 1
            start_pad = sequence_length - actual_seq_len
            adjusted_tensor[env_id, start_pad:, :] = input_tensor[env_id, :actual_seq_len, :]
        else:
            # If the sequence is longer, take the most recent `sequence_length` entries
            adjusted_tensor[env_id, :, :] = input_tensor[env_id, actual_seq_len - sequence_length:actual_seq_len, :]

    return adjusted_tensor


class DataNormalizer:
    def __init__(self, cfg, file_list, save_path=None):
        self.cfg = cfg
        self.normalize_keys = self.cfg.train.normalize_keys
        self.normalization_path = self.cfg.train.normalize_file if self.cfg.train.load_stats else save_path + '/normalization.pkl'
        self.normalize_dict = {"mean": {}, "std": {}}
        self.file_list = file_list
        self.remove_failed_trajectories()

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
        if self.cfg.train.load_stats and os.path.exists(self.normalization_path):
            if os.path.exists(self.normalization_path):
                with open(self.normalization_path, 'rb') as f:
                    self.normalize_dict = pickle.load(f)
                    print('Loaded stats file from: ', self.normalization_path)
            else:
                raise FileNotFoundError('Failed to load stats file')
        else:
            self.create_normalization_file()

    def create_normalization_file(self):
        """Create a new normalization file."""
        for norm_key in self.normalize_keys:
            print(f'Creating new normalization file for {norm_key}')
            data = self.aggregate_data(norm_key)
            # Pass the correction argument here if needed
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
        if norm_key == 'plug_hand_pos':
            if data.shape[1] == 7:  # Combined position and quaternion
                # Assuming the first 3 columns are positions and the last 4 columns are quaternions
                pos = data[:, :3]
                quat = data[:, 3:]

                # Process position data
                diff_pos = pos - pos[0, :]
                self.normalize_dict['mean']["plug_hand_pos"] = np.mean(pos, axis=0)
                self.normalize_dict['std']["plug_hand_pos"] = np.std(pos, axis=0)
                self.normalize_dict['mean']["plug_hand_pos_diff"] = np.mean(diff_pos, axis=0)
                self.normalize_dict['std']["plug_hand_pos_diff"] = np.std(diff_pos, axis=0)

                # Process quaternion data
                self.normalize_dict['mean']["plug_hand_quat"] = np.mean(quat, axis=0)
                self.normalize_dict['std']["plug_hand_quat"] = np.std(quat, axis=0)
                euler = Rotation.from_quat(quat).as_euler('xyz')
                self.normalize_dict['mean']["plug_hand_euler"] = np.mean(euler, axis=0)
                self.normalize_dict['std']["plug_hand_euler"] = np.std(euler, axis=0)

                diff_euler = euler - euler[0, :]
                self.normalize_dict['mean']["plug_hand_diff_euler"] = np.mean(diff_euler, axis=0)
                self.normalize_dict['std']["plug_hand_diff_euler"] = np.std(diff_euler, axis=0)

                sin_cos_repr = np.hstack((np.sin(euler[:, 0:1]), np.cos(euler[:, 0:1]),
                                          np.sin(euler[:, 1:2]), np.cos(euler[:, 1:2])))

                self.normalize_dict['mean']["plug_hand_sin_cos_euler"] = np.mean(sin_cos_repr, axis=0)
                self.normalize_dict['std']["plug_hand_sin_cos_euler"] = np.std(sin_cos_repr, axis=0)
            else:  # Only position data
                pos = data
                diff_pos = pos - pos[0, :]
                self.normalize_dict['mean']["plug_hand_pos"] = np.mean(pos, axis=0)
                self.normalize_dict['std']["plug_hand_pos"] = np.std(pos, axis=0)
                self.normalize_dict['mean']["plug_hand_pos_diff"] = np.mean(diff_pos, axis=0)
                self.normalize_dict['std']["plug_hand_pos_diff"] = np.std(diff_pos, axis=0)

        elif norm_key == 'plug_hand_quat':
            self.normalize_dict['mean'][norm_key] = np.mean(data, axis=0)
            self.normalize_dict['std'][norm_key] = np.std(data, axis=0)

            euler = Rotation.from_quat(data).as_euler('xyz')
            self.normalize_dict['mean']["plug_hand_euler"] = np.mean(euler, axis=0)
            self.normalize_dict['std']["plug_hand_euler"] = np.std(euler, axis=0)

            diff_euler = euler - euler[0, :]

            self.normalize_dict['mean']["plug_hand_diff_euler"] = np.mean(diff_euler, axis=0)
            self.normalize_dict['std']["plug_hand_diff_euler"] = np.std(diff_euler, axis=0)

            sin_cos_repr = np.hstack((np.sin(euler[:, 0:1]), np.cos(euler[:, 0:1]),
                                      np.sin(euler[:, 1:2]), np.cos(euler[:, 1:2])))

            self.normalize_dict['mean']["plug_hand_sin_cos_euler"] = np.mean(sin_cos_repr, axis=0)
            self.normalize_dict['std']["plug_hand_sin_cos_euler"] = np.std(sin_cos_repr, axis=0)

        else:
            # Handle other normalization keys
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

class TactileDataset(Dataset):
    def __init__(self, files, sequence_length=500,
                 full_sequence=False,
                 normalize_dict=None,
                 stride=5,
                 img_transform=None,
                 tactile_transform=None,
                 tactile_channel=3,
                 include_img=True,
                 include_lin=True,
                 include_tactile=True
                 ):

        self.all_folders = files
        self.sequence_length = sequence_length
        self.stride = stride  # sequence_length
        self.full_sequence = full_sequence
        self.normalize_dict = normalize_dict

        self.include_img = include_img
        self.include_lin = include_lin
        self.include_tactile = include_tactile

        self.tactile_channel = tactile_channel
        if self.tactile_channel == 1:
            self.to_gray = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()  # Convert PIL Image back to tensor
            ])

        self.to_torch = lambda x: torch.from_numpy(x).float()

        self.img_transform = img_transform
        self.tactile_transform = tactile_transform

        # Store indices corresponding to each trajectory
        self.indices_per_trajectory = []
        for file_idx, file in enumerate(self.all_folders):
            data = np.load(file)
            done = data["done"]
            done_idx = done.nonzero()[0][-1]
            total_len = done_idx
            if self.full_sequence:
                assert total_len == sequence_length, f"Sequence length mismatch in {file}."
                self.indices_per_trajectory.append((file_idx, 0))
            else:
                if total_len >= self.sequence_length:
                    num_subsequences = (total_len - self.sequence_length) // self.stride + 1
                    self.indices_per_trajectory.extend([(file_idx, i * self.stride) for i in range(num_subsequences)])
        print('Total sub trajectories:', len(self.indices_per_trajectory))

    def __len__(self):
        return int((len(self.indices_per_trajectory)))

    def extract_sequence(self, data, key, start_idx):
        # Extract a sequence of specific length from the array
        return data[key][start_idx:start_idx + self.sequence_length]

    def __getitem__(self, idx, diff_tac=True, diff=True):
        file_idx, start_idx = self.indices_per_trajectory[idx]
        file_path = self.all_folders[file_idx]

        # Load data from the file
        data = np.load(file_path)
        tactile_folder = file_path[:-7].replace('obs', 'tactile')
        img_folder = file_path[:-7].replace('obs', 'img')

        done = data["done"]
        done_idx = done.nonzero()[0][-1]

        # Mask generation (for varied sequence lengths)
        padding_length = max(0, self.sequence_length - (done_idx - start_idx))
        mask = np.ones(self.sequence_length)
        mask[:padding_length] = 0

        keys = ["eef_pos", "action", "latent", "obs_hist", "noisy_socket_pos",
                "hand_joints", "plug_hand_quat", "plug_hand_pos",
                "plug_pos_error", "plug_quat_error"
                ]

        data_seq = {key: self.extract_sequence(data, key, start_idx) for key in keys}

        # Tactile input [T F W H C]
        if self.include_tactile:
            tactile_input = [np.load(os.path.join(tactile_folder, f'tactile_{i}.npz'))['tactile'] for i in
                             range(start_idx, start_idx + self.sequence_length)]
            if diff_tac:
                first_tactile = np.load(os.path.join(tactile_folder, f'tactile_1.npz'))['tactile']
                tactile_input = [(tac - first_tactile) + 1e-6 for tac in tactile_input]

            tactile_input = self.to_torch(np.stack(tactile_input))
            T, F, W, H, C = tactile_input.shape
            if self.tactile_transform is not None:
                tactile_input_reshaped = tactile_input.view(-1, 3, *tactile_input.shape[-2:])
                if self.tactile_channel == 1:
                    tactile_input_reshaped = torch.stack([self.to_gray(image) for image in tactile_input_reshaped])

                tactile_input = self.tactile_transform(tactile_input_reshaped)
                tactile_input = tactile_input.view(T, F, self.tactile_channel, *tactile_input.shape[-2:])
        else:
            tactile_input = torch.zeros(1)

        if self.include_img:
            img_input = np.stack([np.load(os.path.join(img_folder, f'img_{i}.npz'))['img'] for i in
                                  range(start_idx, start_idx + self.sequence_length)])

            if self.img_transform is not None:
                img_input = self.img_transform(self.to_torch(img_input))
        else:
            img_input = torch.zeros(1)

        eef_pos = data_seq["eef_pos"]
        action = data_seq["action"]
        obs_hist = data_seq["obs_hist"]
        noisy_socket_pos = data_seq["noisy_socket_pos"][:, :3]
        euler = Rotation.from_quat(data_seq["plug_hand_quat"]).as_euler('xyz')
        plug_hand_pos = data_seq["plug_hand_pos"]

        # plug_pos_error = data_seq["plug_pos_error"]
        # plug_quat_error = data_seq["plug_quat_error"]

        latent = data_seq["latent"]

        if diff:
            euler = euler - Rotation.from_quat(data["plug_hand_quat"][0, :]).as_euler('xyz')
            plug_hand_pos = plug_hand_pos - data["plug_hand_pos"][0, :]

        # Normalizing
        if self.normalize_dict is not None:
            eef_pos = (eef_pos - self.normalize_dict["mean"]["eef_pos"]) / self.normalize_dict["std"]["eef_pos"]
            noisy_socket_pos = (noisy_socket_pos - self.normalize_dict["mean"]["noisy_socket_pos"][:3]) / self.normalize_dict["std"]["noisy_socket_pos"][:3]
            if not diff:
                euler = (euler - self.normalize_dict["mean"]["plug_hand_euler"]) / self.normalize_dict["std"]["plug_hand_euler"]
                plug_hand_pos = (plug_hand_pos - self.normalize_dict["mean"]["plug_hand_pos"]) / self.normalize_dict["std"]["plug_hand_pos"]
            else:
                euler = (euler - self.normalize_dict["mean"]["plug_hand_diff_euler"]) / self.normalize_dict["std"]["plug_hand_diff_euler"]
                plug_hand_pos = (plug_hand_pos - self.normalize_dict["mean"]["plug_hand_pos_diff"]) / self.normalize_dict["std"]["plug_hand_pos_diff"]

        obj_pos_rpy = np.hstack((plug_hand_pos, euler))

        label = latent
        shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)

        # Generate and add noise to noisy_socket_pos and obj_pos_rpy
        noisy_socket_pos_noise = np.random.normal(loc=0, scale=0.002, size=noisy_socket_pos.shape)
        obj_pos_rpy_noise = np.random.normal(loc=0, scale=0.002, size=obj_pos_rpy.shape)
        noisy_socket_pos += noisy_socket_pos_noise

        lin_input = np.concatenate([
            eef_pos,      # 12
            noisy_socket_pos + noisy_socket_pos_noise, # 3
            action, # 6
            obj_pos_rpy + obj_pos_rpy_noise, # 6
        ], axis=-1)

        # Convert to torch tensors
        tensors = [tensor if isinstance(tensor, torch.Tensor) else self.to_torch(tensor) for tensor in [tactile_input,
                                                                                                        img_input,
                                                                                                        lin_input,
                                                                                                        obj_pos_rpy,
                                                                                                        obs_hist,
                                                                                                        label,
                                                                                                        action,
                                                                                                        mask]]

        return tuple(tensors)


class TactileTestDataset(Dataset):
    def __init__(self, files,
                 sequence_length=500,
                 normalize_dict=None,
                 stride=3,
                 img_transform=None,
                 tactile_transform=None,
                 tactile_channel=3,
                 real_data=False
                 ):

        self.all_folders = files
        self.sequence_length = sequence_length
        self.stride = stride  # sequence_length
        self.normalize_dict = normalize_dict
        self.real_data = real_data
        self.tactile_channel = tactile_channel

        if self.tactile_channel == 1:
            self.to_gray = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()  # Convert PIL Image back to tensor
            ])

        self.to_torch = lambda x: torch.from_numpy(x).float()

        self.img_transform = img_transform
        self.tactile_transform = tactile_transform

        # Store indices corresponding to each trajectory
        self.indices_per_trajectory = []
        for file_idx, file in enumerate(self.all_folders):
            data = np.load(file)
            done = data["done"]
            done_idx = done.nonzero()[0][-1]
            total_len = done_idx
            if total_len >= self.sequence_length:
                num_subsequences = (total_len - self.sequence_length - self.stride) // self.stride + 1
                self.indices_per_trajectory.extend(
                    [(file_idx, self.stride + i * self.stride) for i in range(num_subsequences)])
        print('Total sub trajectories:', len(self.indices_per_trajectory))

    def __len__(self):
        return int((len(self.indices_per_trajectory)))

    def extract_sequence(self, data, key, start_idx):
        # Extract a sequence of specific length from the array
        return data[key][start_idx:start_idx + self.sequence_length]

    def __getitem__(self, idx, diff_tac=True, diff_pos=True, with_img=False):
        file_idx, start_idx = self.indices_per_trajectory[idx]
        file_path = self.all_folders[file_idx]
        tactile_folder = file_path[:-7].replace('obs', 'tactile')
        img_folder = file_path[:-7].replace('obs', 'img')

        # Load data from the file
        data = np.load(file_path)
        data_dict = {key: data[key] for key in data}
        if self.real_data:
            data_dict["plug_hand_quat"] = data_dict["plug_hand_pos"][:, 3:]
            data_dict["plug_hand_pos"] = data_dict["plug_hand_pos"][:, :3]

        keys = ["eef_pos", "action", "latent", "plug_hand_pos", "plug_hand_quat"]
        data_seq = {key: self.extract_sequence(data_dict, key, start_idx) for key in keys}

        tactile_input = [np.load(os.path.join(tactile_folder, f'tactile_{i}.npz'))['tactile'] for i in
                         range(start_idx, start_idx + self.sequence_length)]
        if diff_tac:
            first_tactile = np.load(os.path.join(tactile_folder, f'tactile_1.npz'))['tactile']
            tactile_input = [(tac - first_tactile) + 1e-6 for tac in tactile_input]

        tactile_input = self.to_torch(np.stack(tactile_input))
        T, F, C, W, H = tactile_input.shape

        if self.tactile_transform is not None:
            tactile_input_reshaped = tactile_input.view(-1, C, *tactile_input.shape[-2:])
            if self.tactile_channel == 1:
                tactile_input_reshaped = torch.stack([self.to_gray(image) for image in tactile_input_reshaped])

            tactile_input = self.tactile_transform(tactile_input_reshaped)
            tactile_input = tactile_input.view(T, F, self.tactile_channel, *tactile_input.shape[-2:])

        if with_img:
            img_input = [np.load(os.path.join(img_folder, f'img_{i}.npz'))['img'] for i in
                         range(start_idx, start_idx + self.sequence_length)]
            if diff_tac:
                first_img = np.load(os.path.join(img_folder, f'img_1.npz'))['img']
                img_input = [(im - first_img) + 1e-6 for im in img_input]

            img_input = np.stack(img_input)
            if self.img_transform is not None:
                img_input = self.img_transform(self.to_torch(img_input))
        else:
            img_input = np.zeros_like(data_seq["eef_pos"])

        eef_pos = data_seq["eef_pos"]
        plug_hand_pos = data_seq["plug_hand_pos"]
        euler = Rotation.from_quat(data_seq["plug_hand_quat"]).as_euler('xyz')

        if diff_pos:
            euler = euler - Rotation.from_quat(data_dict["plug_hand_quat"][0, :]).as_euler('xyz')
            plug_hand_pos = plug_hand_pos - data_dict["plug_hand_pos"][0, :]

        # Normalizing
        if self.normalize_dict is not None:
            eef_pos = (eef_pos - self.normalize_dict["mean"]["eef_pos"]) / self.normalize_dict["std"]["eef_pos"]

            if not diff_pos:
                euler = (euler - self.normalize_dict["mean"]["plug_hand_euler"]) / self.normalize_dict["std"][
                    "plug_hand_euler"]
                plug_hand_pos = (plug_hand_pos - self.normalize_dict["mean"]["plug_hand_pos"]) / \
                                self.normalize_dict["std"]["plug_hand_pos"]

            else:
                euler = (euler - self.normalize_dict["mean"]["plug_hand_diff_euler"]) / self.normalize_dict["std"][
                    "plug_hand_diff_euler"]
                plug_hand_pos = (plug_hand_pos - self.normalize_dict["mean"]["plug_hand_pos_diff"]) / \
                                self.normalize_dict["std"]["plug_hand_pos_diff"]

        label = np.hstack((plug_hand_pos, euler))

        # Output
        # shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)

        lin_input = np.concatenate([
            eef_pos,
            # noisy_socket_pos,
            # hand_joints,
        ], axis=-1)

        # Convert to torch tensors
        tensors = [tensor if isinstance(tensor, torch.Tensor) else self.to_torch(tensor) for tensor in [tactile_input,
                                                                                                        img_input,
                                                                                                        lin_input,
                                                                                                        label]]

        return tuple(tensors)


class TactileRealDataset(Dataset):
    def __init__(self, files, sequence_length=500, full_sequence=False, normalize_dict=None):

        self.all_folders = files
        self.sequence_length = sequence_length
        self.full_sequence = full_sequence
        self.normalize_dict = normalize_dict

        self.to_torch = lambda x: torch.from_numpy(x).float()

        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, ), (0.5, )),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return int((len(self.all_folders)))

    def __getitem__(self, idx):

        data = np.load(self.all_folders[idx])

        # cnn input
        tactile = data["tactile"]
        left_finger = tactile[:, 0, ...]  # no one likes this ... dhruv!
        right_finger = tactile[:, 1, ...]
        bottom_finger = tactile[:, 2, ...]

        # linear input
        eef_pos = data['eef_pos']
        action = data["action"]
        latent = data["latent"]
        obs_hist = data["obs_hist"]
        done = data["done"]

        # socket_pos = data["socket_pos"]
        # plug_pos = data["plug_pos"]
        # target = data["target"]
        # priv_obs = data["priv_obs"]
        # contacts = data["contacts"]
        # ft = data["ft"]

        # Normalizing inputs
        if self.normalize_dict is not None:
            eef_pos = (eef_pos - self.normalize_dict["mean"]["eef_pos"]) / self.normalize_dict["std"]["eef_pos"]

        # output providing a_{t-1} to input
        shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)

        lin_input = np.concatenate([eef_pos, shift_action_right], axis=-1)

        done_idx = done.nonzero()[0][-1]

        # doing these operations to enable transform. They have no meaning if written separately.
        left_finger = self.transform(self.to_torch(left_finger).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        right_finger = self.transform(self.to_torch(right_finger).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        bottom_finger = self.transform(self.to_torch(bottom_finger).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()

        # TODO: here we can specify different label
        # latent = priv_obs[:, -3:]  # change here for supervised

        if self.full_sequence:
            mask = np.zeros_like(done)
            mask[:done_idx] = 1
            mask = mask.astype(bool)
            return ((self.to_torch(left_finger),
                     self.to_torch(right_finger),
                     self.to_torch(bottom_finger)),
                    self.to_torch(lin_input),
                    self.to_torch(obs_hist),
                    self.to_torch(latent),
                    self.to_torch(action),
                    self.to_torch(mask))

        # if full_sequence is False
        # we find a random index, and then take the previous sequence_length number of frames from that index
        # if the frames before that index are less than sequence_length, we pad with zeros

        # generating mask (for varied sequence lengths)
        end_idx = np.random.randint(0, done_idx)
        start_idx = max(0, end_idx - self.sequence_length)
        padding_length = self.sequence_length - (end_idx - start_idx)

        # print("start_idx: ", start_idx, "end_idx: ", end_idx, "padding_length: ", padding_length)
        build_traj = lambda x: np.concatenate([np.zeros((padding_length, *x.shape[1:])), x[start_idx: end_idx, ...]],
                                              axis=0)

        mask = np.ones(self.sequence_length)
        mask[:padding_length] = 0
        mask = mask.astype(bool)

        left_finger = build_traj(left_finger)
        right_finger = build_traj(right_finger)
        bottom_finger = build_traj(bottom_finger)
        lin_input = build_traj(lin_input)

        action = build_traj(action)
        latent = build_traj(latent)
        obs_hist = build_traj(obs_hist)
        # priv_obs = build_traj(priv_obs)

        # converting to torch tensors
        left_finger = self.to_torch(left_finger)
        right_finger = self.to_torch(right_finger)
        bottom_finger = self.to_torch(bottom_finger)
        lin_input = self.to_torch(lin_input)
        obs_hist = self.to_torch(obs_hist)
        latent = self.to_torch(latent)
        action = self.to_torch(action)
        mask = self.to_torch(mask)
        # priv_obs = self.to_torch(priv_obs)

        return (left_finger, right_finger, bottom_finger), lin_input, obs_hist, latent, action, mask


# # for tests
if __name__ == "__main__":
    files = glob("/common/users/dm1487/inhand_manipulation_data_store/*/*/*.npz")
    ds = TactileDataset(files, sequence_length=100, full_sequence=False)

    cnn_input, lin_input, obs_hist, latent, action, mask = next(iter(ds))
    print(obs_hist.shape)
    print(action.shape)
    print(lin_input.shape)
    print(latent.shape)
    print(mask.shape)
