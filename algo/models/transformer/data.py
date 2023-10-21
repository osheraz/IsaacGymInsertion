from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

class TactileDataset(Dataset):
    def __init__(self, files, sequence_length):

        self.all_folders = files
        self.sequence_length = int(sequence_length)
        
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

        # generating mask (for varied sequence lengths)
        done = data["done"]
        done_idx = done.nonzero()[0][-1]
        mask = np.zeros_like(done)
        mask[:done_idx] = 1
        mask = mask.astype(bool)

        cnn_input = tactile
        lin_input = np.concatenate([arm_joints, eef_pos, noisy_socket_pos, action, target], axis=-1)

        return cnn_input, lin_input, latent, mask