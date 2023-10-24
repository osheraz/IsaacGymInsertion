from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

class TactileDataset(Dataset):
    def __init__(self, files, sequence_length=500):

        self.all_folders = files
        
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):
        data = np.load(self.all_folders[idx])
        
        # cnn input
        tactile = data["tactile"]

        # linear input
        arm_joints = data["arm_joints"]
        eef_pos = data["eef_pos"]
        noisy_socket_pos = data["noisy_socket_pos"]
        action = data["action"]
        target = data["target"]

        # output
        latent = data["latent"]

        obs_hist = data["obs_hist"]
        priv_obs = data["priv_obs"]


        # generating mask (for varied sequence lengths)
        done = data["done"]
        done_idx = done.nonzero()[0][-1]
        mask = np.zeros_like(done)
        mask[:done_idx] = 1
        mask = mask.astype(bool)

        cnn_input = tactile
        cnn_input = np.concatenate([cnn_input[:, 0, ...], cnn_input[:, 1, ...], cnn_input[:, 2, ...]], axis=-1)

        # providing a_{t-1} to input
        shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)
        shift_target_right = np.concatenate([np.zeros((1, action.shape[-1])), target[:-1, :]], axis=0)

        lin_input = np.concatenate([arm_joints, eef_pos, noisy_socket_pos, shift_action_right, shift_target_right], axis=-1)

        # converting to torch tensors
        cnn_input = torch.from_numpy(cnn_input).float()
        lin_input = torch.from_numpy(lin_input).float()
        obs_hist = torch.from_numpy(obs_hist).float()
        # priv_obs = torch.from_numpy(priv_obs).float()
        latent = torch.from_numpy(latent).float()
        action = torch.from_numpy(action).float()
        mask = torch.from_numpy(mask).float()

        return cnn_input, lin_input, obs_hist, latent, action, mask

# for tests   
# if __name__ == "__main__":

#     files = glob("/common/users/dm1487/inhand_manipulation_data_store/*/*/*.npz")
#     ds = TactileDataset(files, 500)
    
#     cnn_input, lin_input, latent, mask = next(iter(ds))
#     print(cnn_input.shape)
#     print(lin_input.shape)
#     print(latent.shape)
#     print(mask.shape)