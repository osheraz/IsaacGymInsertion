from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch


class TactileDataset(Dataset):
    def __init__(self, files, sequence_length=500, full_sequence=False, normalize_dict=None, stride = 10):

        self.all_folders = files
        self.sequence_length = sequence_length
        self.stride = sequence_length
        self.full_sequence = full_sequence
        self.normalize_dict = normalize_dict

        self.to_torch = lambda x: torch.from_numpy(x).float()

        # Store indices corresponding to each trajectory
        self.indices_per_trajectory = []
        for file_idx, file in enumerate(self.all_folders):
            data = np.load(file)
            total_len = len(data["action"])
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

    def __getitem__(self, idx):
        file_idx, start_idx = self.indices_per_trajectory[idx]
        file_path = self.all_folders[file_idx]

        # Load data from the file
        data = np.load(file_path)

        done = data["done"]
        done_idx = done.nonzero()[0][-1]

        # Mask generation (for varied sequence lengths)
        padding_length = max(0, self.sequence_length - (done_idx - start_idx))
        mask = np.ones(self.sequence_length)
        mask[:padding_length] = 0

        keys = ["tactile", "eef_pos", "action", "latent", "obs_hist", "contacts", "hand_joints"]
        data = {key: self.extract_sequence(data, key, start_idx) for key in keys}

        # Tactile input [T F C W H]
        # left_finger, right_finger, bottom_finger = [data["tactile"][:, i, ...] for i in range(3)]

        #  For student predictions
        tactile_input = data["tactile"]
        T, F, W, H, C = tactile_input.shape
        tactile_input = tactile_input.reshape(T, F*C, W, H)
        eef_pos = data["eef_pos"]
        hand_joints = data["hand_joints"]
        action = data["action"]
        contacts = data["action"] # data["contacts"]
        # For teacher predictions - will be normalized by the model
        obs_hist = data["obs_hist"]
        latent = data["latent"]

        # Normalizing inputs
        if self.normalize_dict is not None:
            eef_pos = (eef_pos - self.normalize_dict["mean"]["eef_pos"]) / self.normalize_dict["std"]["eef_pos"]
            hand_joints = (hand_joints - self.normalize_dict["mean"]["hand_joints"]) / self.normalize_dict["std"]["hand_joints"]

        # Output
        shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)
        lin_input = np.concatenate([eef_pos, hand_joints, shift_action_right], axis=-1)

        # Convert to torch tensors
        tensors = [self.to_torch(tensor) for tensor in [tactile_input,
                                                        lin_input,
                                                        contacts,
                                                        obs_hist,
                                                        latent,
                                                        action,
                                                        mask]]

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
        end_idx = np.random.randint(0, done_idx)  # 0 and 500
        start_idx = max(0, end_idx - self.sequence_length)  # max(0, 60 - 50) = 10 -> 60
        padding_length = self.sequence_length - (end_idx - start_idx)  # 50 - (60 - 10) = 0 -> 50

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
