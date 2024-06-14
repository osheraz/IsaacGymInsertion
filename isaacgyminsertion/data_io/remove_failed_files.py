import os
import numpy as np
from glob import glob
import re
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def check_and_crop_large_files(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'img' in dirs and 'tactile' in dirs:
            img_file_list = sorted(glob(os.path.join(root, 'img', '*.npz')), key=natural_sort_key)
            tactile_file_list = sorted(glob(os.path.join(root, 'tactile', '*.npz')), key=natural_sort_key)
            obs_file = os.path.join(root, 'obs', 'obs.npz')

            if len(img_file_list) > 500:
                print(f'Cropping sequences in {root} due to excessive length')

                # Remove excess img files
                for img_file in img_file_list[500:]:
                    os.remove(img_file)
                    print(f'Removing  {img_file} due to excessive length')

            if len(tactile_file_list) > 500:
                # Remove excess tactile files
                for tactile_file in tactile_file_list[500:]:
                    os.remove(tactile_file)
                    print(f'Removing  {tactile_file} due to excessive length')

                # Crop the obs.npz file if it exists
            if len(img_file_list) > 500 or len(tactile_file_list) > 500:
                if os.path.exists(obs_file):
                    data = np.load(obs_file)
                    cropped_data = {key: (data[key][:500] if data[key].shape[0] > 500 else data[key]) for key in
                                    data.files}
                    np.savez_compressed(obs_file, **cropped_data)


# Define the path to the processed data
processed_data_path = "/home/roblab20/tactile_diffusion/datastore_real_v1"

# Check and crop directories with sequences longer than 500
check_and_crop_large_files(processed_data_path)
