#%%
import numpy as np
from glob import glob
import os
#%%
assem = 'round_peg_hole_16mm_loose_noise'
data_folders = glob(f'../outputs/debug/{assem}/*.npz')
#%%
total_grasps = 0
for i in data_folders:
    data = np.load(i)
    num_grasps = data['socket_pos'].shape[0]
    total_grasps += num_grasps
#%%
socket_pos = np.zeros((total_grasps, 3))
socket_quat = np.zeros((total_grasps, 4))
plug_pos = np.zeros((total_grasps, 3))
plug_quat = np.zeros((total_grasps, 4))
dof_pos = np.zeros((total_grasps, 15))
#%%
last_ptr = 0
for i in data_folders:
    data = np.load(i)
    num_grasps = data['socket_pos'].shape[0]
    # print(data['socket_pos'].shape, np.squeeze(data['socket_pos'], axis=1).shape)
    socket_pos[last_ptr: last_ptr+num_grasps] = np.squeeze(data['socket_pos'], axis=1)
    socket_quat[last_ptr: last_ptr+num_grasps] = np.squeeze(data['socket_quat'], axis=1)
    plug_pos[last_ptr: last_ptr+num_grasps] = np.squeeze(data['plug_pos'], axis=1)
    plug_quat[last_ptr: last_ptr+num_grasps] = np.squeeze(data['plug_quat'], axis=1)
    dof_pos[last_ptr: last_ptr+num_grasps] = np.squeeze(data['dof_pos'], axis=1)
    last_ptr += num_grasps

print(f'saving {last_ptr} grasping poses: ', assem)
#%%
final_grasps_folder = 'initial_grasp_data'
np.savez_compressed(f'{assem}.npz', socket_pos=socket_pos, socket_quat=socket_quat, plug_pos=plug_pos, plug_quat=plug_quat, dof_pos=dof_pos)
#%%
