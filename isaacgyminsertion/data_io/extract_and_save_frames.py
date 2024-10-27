import os
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(file_path):
    # Define output folders
    output_img_folder = os.path.join(file_path[:-4], 'img')
    output_rgb_folder = os.path.join(file_path[:-4], 'rgb')
    output_seg_folder = os.path.join(file_path[:-4], 'seg')
    output_tactile_folder = os.path.join(file_path[:-4], 'tactile')
    output_pcl_folder = os.path.join(file_path[:-4], 'pcl')
    output_obs_folder = os.path.join(file_path[:-4], 'obs')

    # Create directories if they do not exist
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_seg_folder, exist_ok=True)
    os.makedirs(output_obs_folder, exist_ok=True)
    os.makedirs(output_tactile_folder, exist_ok=True)
    os.makedirs(output_pcl_folder, exist_ok=True)
    os.makedirs(output_rgb_folder, exist_ok=True)

    print(f'Processing {file_path}')

    # Load the .npz file
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as exc:
        print(f'File processing generated an exception: {exc}')
        os.remove(file_path)
        # print(f'Check {file_path}')
        return

    # Extract img and tactile sequences if they exist
    done_idx = data['done'].nonzero()[-1][0]
    img_sequence = data['img'][:done_idx, :] if 'img' in data else None
    seg_sequence = data['seg'][:done_idx, :] if 'seg' in data else None
    pcl_sequence = data['pcl'][:done_idx, :] if 'pcl' in data else None
    tactile_sequence = data['tactile'][:done_idx, :] if 'tactile' in data else None
    rgb_sequence = data['rgb'][:done_idx, :] if 'rgb' in data else None

    # Remove img and tactile from the data dictionary
    data_dict = {key: data[key][:done_idx + 1] for key in data.files if key not in ['img', 'rgb', 'tactile', 'seg', 'pcl']}

    # Save each frame in the img sequence if it exists
    if img_sequence is not None:
        for idx, img in enumerate(img_sequence):
            img_filename = os.path.join(output_img_folder, f"img_{idx}.npz")
            np.savez_compressed(img_filename, img=img)

    if seg_sequence is not None:
        for idx, seg in enumerate(seg_sequence):
            seg_filename = os.path.join(output_seg_folder, f"seg_{idx}.npz")
            np.savez_compressed(seg_filename, seg=seg)

    if rgb_sequence is not None:
        for idx, rgb in enumerate(rgb_sequence):
            rgb_filename = os.path.join(output_rgb_folder, f"rgb_{idx}.npz")
            np.savez_compressed(rgb_filename, rgb=rgb)

    if pcl_sequence is not None:
        for idx, pcl in enumerate(pcl_sequence):
            pcl_filename = os.path.join(output_pcl_folder, f"pcl_{idx}.npz")
            np.savez_compressed(pcl_filename, pcl=pcl)

    # Save each frame in the tactile sequence if it exists
    if tactile_sequence is not None:
        for idx, tactile in enumerate(tactile_sequence):
            tactile_filename = os.path.join(output_tactile_folder, f"tactile_{idx}.npz")
            np.savez_compressed(tactile_filename, tactile=tactile)

    # Save the remaining data to a new .npz file
    remaining_data_filename = os.path.join(output_obs_folder, f"obs.npz")
    np.savez_compressed(remaining_data_filename, **data_dict)

    # Remove the original .npz file
    os.remove(file_path)
    print(f'{file_path} has been removed')


def extract_and_save_frames(file_list):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path) for file_path in file_list]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'File processing generated an exception: {exc}')


# Define the path to the data
# data_path = "/home/roblab20/tactile_tests/third"
# data_path = "/home/osher/tactile_insertion/datastore_42_gt_test"
# data_path = "/home/osher/tactile_insertion/datastore_42_teacher"
data_path = "/home/roblab20/for_paper/datastore_real"

print('Loading trajectories from', data_path)

# Get a list of all .npz files in the data path
traj_list = glob(os.path.join(data_path, '*/*.npz'))

# Extract and save frames
extract_and_save_frames(traj_list)
