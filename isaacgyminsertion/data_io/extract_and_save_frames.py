import os
import numpy as np
from glob import glob

def extract_and_save_frames(file_list):
    for num, file_path in enumerate(file_list):
        # Define output folders
        output_img_folder = os.path.join(file_path[:-4], 'img')
        output_tactile_folder = os.path.join(file_path[:-4], 'tactile')
        output_obs_folder = os.path.join(file_path[:-4], 'obs')

        # Create directories if they do not exist
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_obs_folder, exist_ok=True)
        os.makedirs(output_tactile_folder, exist_ok=True)

        print(f'Processing {file_path} ({num + 1}/{len(file_list)})')

        # Load the .npz file
        data = np.load(file_path)

        # Extract img and tactile sequences
        img_sequence = data['img']
        tactile_sequence = data['tactile']

        # Remove img and tactile from the data dictionary
        data_dict = {key: data[key] for key in data.files if key not in ['img', 'tactile']}

        # Save each frame in the img sequence
        for idx, img in enumerate(img_sequence):
            img_filename = os.path.join(output_img_folder, f"img_{idx}.npz")
            np.savez_compressed(img_filename, img=img)

        # Save each frame in the tactile sequence
        for idx, tactile in enumerate(tactile_sequence):
            tactile_filename = os.path.join(output_tactile_folder, f"tactile_{idx}.npz")
            np.savez_compressed(tactile_filename, tactile=tactile)

        # Save the remaining data to a new .npz file
        remaining_data_filename = os.path.join(output_obs_folder, f"obs.npz")
        np.savez_compressed(remaining_data_filename, **data_dict)

        # Remove the original .npz file
        os.remove(file_path)
        print(f'{file_path} has been removed')

# Define the path to the data
# data_path = "/home/roblab20/tactile_diffusion/datastore_real"
data_path = "/home/roblab20/tactile_insertion/datastore_42_gt_test"

print('Loading trajectories from', data_path)

# Get a list of all .npz files in the data path
traj_list = glob(os.path.join(data_path, '*/*.npz'))

# Extract and save frames
extract_and_save_frames(traj_list)
