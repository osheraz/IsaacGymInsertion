import os
import numpy as np


def extract_and_save_frames(file_list):


    for num, file_path in enumerate(file_list):
        # Load the .npz file
        output_img_folder = os.path.join(file_path[:-4], 'img')
        output_tactile_folder = os.path.join(file_path[:-4], 'tactile')
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_tactile_folder, exist_ok=True)

        print(f'processing {file_path} out of {num/ len(file_list)}')
        data = np.load(file_path)

        # Extract img and tactile sequences
        img_sequence = data['img']
        tactile_sequence = data['tactile']

        # Get the base filename without extension
        base_filename = os.path.basename(file_path).split('.')[0]

        # Iterate over each frame in the img sequence
        for idx, img in enumerate(img_sequence):

            img_filename = os.path.join(output_img_folder, f"{base_filename}_img_{idx}.npz")
            np.savez_compressed(img_filename, img=img)

        # Iterate over each frame in the tactile sequence
        for idx, tactile in enumerate(tactile_sequence):
            tactile_filename = os.path.join(output_tactile_folder, f"{base_filename}_tactile_{idx}.npz")
            np.savez_compressed(tactile_filename, tactile=tactile)


from glob import glob
data_path ="/home/roblab20/tactile_diffusion/datastore_real"
print('Loading trajectories from', data_path)
traj_list = glob(os.path.join(data_path, '*/*.npz'))

# Example usage


extract_and_save_frames(traj_list)
