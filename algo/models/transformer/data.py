from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

class TactileDataset(Dataset):
    def __init__(self, files, sequence_length=500, full_sequence=False):

        self.all_folders = files
        self.sequence_length = sequence_length
        self.full_sequence = full_sequence

        self.to_torch = lambda x: torch.from_numpy(x).float()
        
    def __len__(self):
        return int((len(self.all_folders)))
    
    def __getitem__(self, idx):

        data = np.load(self.all_folders[idx])
        
        # cnn input
        tactile = data["tactile"]
        cnn_input = np.concatenate([tactile[:, 0, ...], tactile[:, 1, ...], tactile[:, 2, ...]], axis=-1)

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

        # providing a_{t-1} to input
        shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)
        shift_target_right = np.concatenate([np.zeros((1, action.shape[-1])), target[:-1, :]], axis=0)

        lin_input = np.concatenate([arm_joints, eef_pos, noisy_socket_pos, shift_action_right, shift_target_right], axis=-1)

        done = data["done"]
        done_idx = done.nonzero()[0][-1]
        
        if self.full_sequence:
            mask = np.zeros_like(done)
            mask[:done_idx] = 1
            mask = mask.astype(bool)
            return self.to_torch(cnn_input), self.to_torch(lin_input), self.to_torch(obs_hist), self.to_torch(latent), self.to_torch(action), self.to_torch(mask)
        
        # if full_sequence is False
        # we find a random index, and then take the previous sequence_length number of frames from that index
        # if the frames before that index are less than sequence_length, we pad with zeros

        # generating mask (for varied sequence lengths)
        end_idx = np.random.randint(0, done_idx) # 0 and 500
        start_idx = max(0, end_idx - self.sequence_length) # max(0, 60 - 50) = 10 -> 60
        padding_length = self.sequence_length - (end_idx - start_idx) # 50 - (60 - 10) = 0 -> 50

        create_data = lambda x: np.concatenate([np.zeros((padding_length, *x.shape[1:])), x[start_idx: end_idx]], axis=0)
        
        mask = np.ones(self.sequence_length)
        mask[:padding_length] = 0
        mask = mask.astype(bool)

        cnn_input = create_data(cnn_input)
        lin_input = create_data(lin_input)
        action = create_data(action)
        latent = create_data(latent)
        obs_hist = create_data(obs_hist)
        # priv_obs = create_data(priv_obs)
        
        # converting to torch tensors
        cnn_input = self.to_torch(cnn_input)
        lin_input = self.to_torch(lin_input)
        obs_hist = self.to_torch(obs_hist)
        latent = self.to_torch(latent)
        action = self.to_torch(action)
        # priv_obs = self.to_torch(priv_obs)
        mask = self.to_torch(mask)

        return cnn_input, lin_input, obs_hist, latent, action, mask

# for tests   
# if __name__ == "__main__":

#     files = glob("/common/users/dm1487/inhand_manipulation_data_store/*/*/*.npz")
#     ds = TactileDataset(files, sequence_length=100, full_sequence=False)
    
#     cnn_input, lin_input, obs_hist, latent, action, mask = next(iter(ds))
#     print(obs_hist.shape)
#     print(action.shape)
#     print(cnn_input.shape)
#     print(lin_input.shape)
#     print(latent.shape)
#     print(mask.shape)