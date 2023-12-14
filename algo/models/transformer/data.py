from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch

class TactileDataset(Dataset):
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
        tactile = data["tactile"][:30]
        # tactile = np.concatenate([tactile[:1, ...], tactile[1:, ...] - tactile[:-1, ...]], axis=0)
        cnn_input_1 = tactile[:, 0, ...][:30]
        cnn_input_2 = tactile[:, 1, ...][:30]
        cnn_input_3 = tactile[:, 2, ...][:30]

        # linear input
        arm_joints = data["arm_joints"][:30]
        eef_pos = data['eef_pos'][:30]
        noisy_socket_pos = data["noisy_socket_pos"][:30][:, :2]
        action = data["action"][:30]
        target = data["target"][:30]
        priv_obs = data["priv_obs"][:30]
        latent = data["latent"][:30]
        obs_hist = data["obs_hist"][:30]

        if self.normalize_dict is not None:
            arm_joints = (arm_joints - self.normalize_dict["mean"]["arm_joints"]) / self.normalize_dict["std"]["arm_joints"]
            eef_pos = (eef_pos - self.normalize_dict["mean"]["eef_pos"]) / self.normalize_dict["std"]["eef_pos"]
            noisy_socket_pos = (noisy_socket_pos - self.normalize_dict["mean"]["noisy_socket_pos"][:2]) / self.normalize_dict["std"]["noisy_socket_pos"][:2]
            target = (target - self.normalize_dict["mean"]["target"]) / self.normalize_dict["std"]["target"]
            # priv_obs = (priv_obs - self.normalize_dict["mean"]["priv_obs"]) / self.normalize_dict["std"]["priv_obs"]

        # output
        # providing a_{t-1} to input
        shift_action_right = np.concatenate([np.zeros((1, action.shape[-1])), action[:-1, :]], axis=0)
        shift_target_right = np.concatenate([np.zeros((1, action.shape[-1])), target[:-1, :]], axis=0)

        lin_input = np.concatenate([eef_pos, shift_action_right, shift_target_right], axis=-1) 

        done = data["done"][:30]
        if done.sum() == 0:
            done_idx = 30
        else:
            done_idx = done.nonzero()[0][-1]


        # doing these operations to enable transform. They have no meaning if written separately.
        # cnn_input_1 = (cnn_input_1 + 0.5)
        # cnn_input_2 = self.to_torch(cnn_input_2).numpy()
        # cnn_input_3 = self.to_torch(cnn_input_3).numpy()

        '''
            plug_hand_pos,   # 3
            plug_hand_quat,  # 4
            physics_params,  # 6
            self.finger_normalized_forces,  # 3
        '''
        latent = data['contacts'][:30]  # change here for supervised

        if self.full_sequence:
            mask = np.zeros_like(done)
            mask[:done_idx] = 1
            mask = mask.astype(bool)
            return ((self.to_torch(cnn_input_1),
                    self.to_torch(cnn_input_2),
                    self.to_torch(cnn_input_3)),
                    self.to_torch(lin_input),
                    self.to_torch(obs_hist),
                    self.to_torch(latent),
                    self.to_torch(action),
                    self.to_torch(mask))
        
        # if full_sequence is False
        # we find a random index, and then take the previous sequence_length number of frames from that index
        # if the frames before that index are less than sequence_length, we pad with zeros

        # generating mask (for varied sequence lengths)
        end_idx = np.random.randint(0, done_idx) # 0 and 500
        start_idx = max(0, end_idx - self.sequence_length) # max(0, 60 - 50) = 10 -> 60
        padding_length = self.sequence_length - (end_idx - start_idx) # 50 - (60 - 10) = 0 -> 50

        # print("start_idx: ", start_idx, "end_idx: ", end_idx, "padding_length: ", padding_length)
        create_data = lambda x: np.concatenate([np.zeros((padding_length, *x.shape[1:])), x[start_idx: end_idx, ...]], axis=0)
        
        mask = np.ones(self.sequence_length)
        mask[:padding_length] = 0
        mask = mask.astype(bool)

        cnn_input_1 = create_data(cnn_input_1)
        cnn_input_2 = create_data(cnn_input_2)
        cnn_input_3 = create_data(cnn_input_3)
        lin_input = create_data(lin_input)
        
        action = create_data(action)
        latent = create_data(latent)
        # plt.plot(latent)
        # plt.show()
        obs_hist = create_data(obs_hist)
        # priv_obs = create_data(priv_obs)
        
        # converting to torch tensors
        cnn_input_1 = self.to_torch(cnn_input_1)
        cnn_input_2 = self.to_torch(cnn_input_2)
        cnn_input_3 = self.to_torch(cnn_input_3)
        lin_input = self.to_torch(lin_input)
        obs_hist = self.to_torch(obs_hist)
        latent = self.to_torch(latent)
        action = self.to_torch(action)
        mask = self.to_torch(mask)
        # priv_obs = self.to_torch(priv_obs)

        return (cnn_input_1, cnn_input_2, cnn_input_3), lin_input, obs_hist, latent, action, mask

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