import numpy as np
import pandas as pd
import os
from glob import glob
import cv2
from vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from scipy import spatial

# Define the parameters
pc_name = os.getlogin()
leds = 'white'
gel = 'clear'
start_sample_time = 2
indenter = ['sphere3']
save = True
sim_data_num = 8
real_data_num = 8
name = f'sim_train_8_paired'
save_name = f'./datasets/data_Allsight/json_data/sim_train_{sim_data_num}_aligned'


# paths of the real and sim dataframes
p_sim = f'./datasets/data_Allsight/json_data/sim_train_{sim_data_num}_transformed.json'
p_real = f'./datasets/data_Allsight/json_data/real_train_{real_data_num}_transformed.json'

# Load real_paths df
df_data_real = pd.read_json(p_real).transpose()
print(df_data_real.shape)
# Load sim_paths df
df_data_sim = pd.read_json(p_sim).transpose()
print(df_data_sim.shape)
# Sample for the real_dataset
# df_data_real = df_data_real[df_data_real.time > start_sample_time]
# df_data_real = df_data_real[df_data_real.num > 1]
if len(df_data_sim)<len(df_data_real):
    n_samples = len(df_data_sim)
    df_data_real = df_data_real.sample(n=n_samples, random_state=42)

# convert to arrays
pose_real = np.array([df_data_real.iloc[idx].pose_transformed[0][:3] for idx in range(df_data_real.shape[0])])
pose_sim = np.array([df_data_sim.iloc[idx].pose_transformed[0] for idx in range(df_data_sim.shape[0])])

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 0.012, 0.016)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
Xc, Yc, Zc = data_for_sphere_along_z(0., 0., 0.012, 0.016)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

# REAL TO SIM MATCHING
tree = spatial.KDTree(pose_sim,leafsize=300)
# _, ind_array = tree.query(pose_real, k=100)
# ind_unique_array = np.array(ind_array)
def find_next_kni(t, ind_to_keep):
    t +=1
    d, ind = tree.query(real_xyz,k=t, distance_upper_bound=10)
    if t < 900:
        if t == 100: 
            yyyy =1
        if t == 1:
            if ind in ind_to_keep:
                ind = find_next_kni(t, ind_to_keep) 
        else:
            if ind[-1] in ind_to_keep:
                ind = find_next_kni(t, ind_to_keep)
                if ind == -1: return -1   
            try: ind = ind[-1]
            except: ind = ind
        return ind  
    else:
        return -1       
        
    
    
ind_to_keep = []

for sample in range(1000):
    t = 0
    real_xyz = pose_real[sample]
    real_image = (cv2.imread(df_data_real['frame'][sample].replace("osher", pc_name))).astype(np.uint8)

    ind = find_next_kni(t, ind_to_keep)
    if ind == -1:
        continue
    # d, ind = tree.query(real_xyz,p=2)
    
    # if ind in ind_to_keep:
    #     continue
    # else:
    ind_to_keep.append(ind)
    sim_xyz = pose_sim[ind]
    sim_image = (cv2.imread(df_data_sim['frame'][ind])).astype(np.uint8)

    ax.scatter(sim_xyz[0], sim_xyz[1], sim_xyz[2], c='black', label='sim')

    cv2.imshow('sim\t\t\treal', np.concatenate((sim_image, real_image), axis=1))
    # ax.scatter(real_xyz[0], real_xyz[1], real_xyz[2], c='red', label='true')

    # print(f'real {real_xyz}\nsim {sim_xyz}\nnorm {np.linalg.norm((real_xyz, sim_xyz)) * 1000}')

    ax.set_xlim((-0.014, 0.014))
    ax.set_ylim((-0.014, 0.014))
    ax.set_zlim((0.0, 0.03))

    # plt.pause(0.001)
    wait = 1 if sample == 0 else 1
    cv2.waitKey(wait) & 0xff

df_data_sim = df_data_sim.iloc[ind_to_keep]
print(df_data_sim.shape)
if save:
    import json
    to_dict = {}
    count = 0
    for index, row in list(df_data_sim.iterrows()):
        to_dict[f"frame_{count}"] = dict(row)
        # to_dict[index] = dict(row)
        count+=1

    with open(r'{}_aligned.json'.format(save_name), 'w') as json_file:
        json.dump(to_dict, json_file, indent=3)

# SIM TO REAL MATCHING
# tree = spatial.KDTree(pose_real)
# for i in range(len(pose_sim)):
#
#     sample = i
#     sim_xyz = pose_sim[sample]
#     sim_image = (cv2.imread(sim_prefix + df_data_sim['frame'][sample])).astype(np.uint8)
#
#     ax.scatter(sim_xyz[0], sim_xyz[1], sim_xyz[2], c='black', label='sim')
#
#     _, ind = tree.query(sim_xyz)
#     real_xyz = pose_real[ind]
#
#     real_image = (cv2.imread(df_data_real['frame'][ind].replace("osher", pc_name))).astype(np.uint8)
#     cp = df_data_real['contact_px'][ind]
#
#     cv2.imshow('sim\t\t\treal', np.concatenate((sim_image, real_image), axis=1))
#
#     ax.scatter(real_xyz[0], real_xyz[1], real_xyz[2], c='red', label='true')
#
#     print(f'real {real_xyz}\nsim {sim_xyz}\nnorm {np.linalg.norm((real_xyz, sim_xyz)) * 1000}')
#
#     ax.set_xlim((-0.014, 0.014))
#     ax.set_ylim((-0.014, 0.014))
#     ax.set_zlim((0.0, 0.03))
#
#     plt.pause(0.001)
#     wait = 1000 if i == 0 else 1000
#     cv2.waitKey(wait) & 0xff