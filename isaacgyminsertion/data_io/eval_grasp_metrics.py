import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class GraspPoseVisualizer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.initial_grasp_poses = {}
        self.total_init_poses = {}
        self.init_socket_pos = {}
        self.init_socket_quat = {}
        self.init_plug_pos = {}
        self.init_plug_quat = {}
        self.init_dof_pos = {}

    def _initialize_grasp_poses(self, subassembly, with_noise=True):
        try:
            sf = subassembly + '_noise' if with_noise else subassembly
            self.initial_grasp_poses[subassembly] = np.load(f'../initial_grasp_data/{sf}.npz')
        except FileNotFoundError as e:
            print(f'Failed to load initial grasp data for {subassembly}. Error: {e}')
            return

        self.total_init_poses[subassembly] = self.initial_grasp_poses[subassembly]['socket_pos'].shape[0]
        self.init_socket_pos[subassembly] = torch.zeros((self.total_init_poses[subassembly], 3))
        self.init_socket_quat[subassembly] = torch.zeros((self.total_init_poses[subassembly], 4))
        self.init_plug_pos[subassembly] = torch.zeros((self.total_init_poses[subassembly], 3))
        self.init_plug_quat[subassembly] = torch.zeros((self.total_init_poses[subassembly], 4))
        self.init_dof_pos[subassembly] = torch.zeros((self.total_init_poses[subassembly], 15))

        socket_pos = self.initial_grasp_poses[subassembly]['socket_pos']
        socket_quat = self.initial_grasp_poses[subassembly]['socket_quat']
        plug_pos = self.initial_grasp_poses[subassembly]['plug_pos']
        plug_quat = self.initial_grasp_poses[subassembly]['plug_quat']
        dof_pos = self.initial_grasp_poses[subassembly]['dof_pos']

        print("Loading Grasping poses for:", subassembly)
        for i in tqdm(range(self.total_init_poses[subassembly])):
            self.init_socket_pos[subassembly][i] = torch.from_numpy(socket_pos[i])
            self.init_socket_quat[subassembly][i] = torch.from_numpy(socket_quat[i])
            self.init_plug_pos[subassembly][i] = torch.from_numpy(plug_pos[i])
            self.init_plug_quat[subassembly][i] = torch.from_numpy(plug_quat[i])
            self.init_dof_pos[subassembly][i] = torch.from_numpy(dof_pos[i])

    def plot_and_save(self, subassembly):
        if subassembly not in self.init_socket_pos:
            print(f"No data loaded for {subassembly}")
            return

        # Ensure the folder exists
        output_folder = f'plots/{subassembly}'
        os.makedirs(output_folder, exist_ok=True)

        # Extract positional data
        socket_positions = self.init_socket_pos[subassembly].numpy()
        plug_positions = self.init_plug_pos[subassembly].numpy()
        num_poses = self.total_init_poses[subassembly]

        # Convert quaternions to roll, pitch, and yaw
        socket_rpy = R.from_quat(self.init_socket_quat[subassembly].numpy()).as_euler('xyz', degrees=True)
        plug_rpy = R.from_quat(self.init_plug_quat[subassembly].numpy()).as_euler('xyz', degrees=True)
        rpy_labels = ['Roll', 'Pitch', 'Yaw']

        # Calculate statistics for RPY angles
        socket_means = np.mean(socket_rpy, axis=0)
        socket_stds = np.std(socket_rpy, axis=0)
        socket_mins = np.min(socket_rpy, axis=0)
        socket_maxs = np.max(socket_rpy, axis=0)

        plug_means = np.mean(plug_rpy, axis=0)
        plug_stds = np.std(plug_rpy, axis=0)
        plug_mins = np.min(plug_rpy, axis=0)
        plug_maxs = np.max(plug_rpy, axis=0)

        # Plot statistics for Roll, Pitch, and Yaw in a single grouped bar chart
        x = np.arange(len(rpy_labels))  # The label locations
        width = 0.2  # The width of the bars

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - 1.5 * width, socket_means, width, label='Socket Mean', color='r')
        ax.bar(x - 0.5 * width, plug_means, width, label='Plug Mean', color='b')
        ax.bar(x + 0.5 * width, socket_stds, width, label='Socket Std', color='r', alpha=0.6)
        ax.bar(x + 1.5 * width, plug_stds, width, label='Plug Std', color='b', alpha=0.6)

        # Adding min and max values as error bars
        ax.errorbar(x - 1.5 * width, socket_means, yerr=[socket_means - socket_mins, socket_maxs - socket_means],
                    fmt='o', color='r', capsize=5)
        ax.errorbar(x - 0.5 * width, plug_means, yerr=[plug_means - plug_mins, plug_maxs - plug_means], fmt='o',
                    color='b', capsize=5)

        # Configure the plot
        ax.set_xlabel('Orientation')
        ax.set_ylabel('Degrees')
        ax.set_title(f'Orientation Statistics for {subassembly}')
        ax.set_xticks(x)
        ax.set_xticklabels(rpy_labels)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{output_folder}/orientation_statistics.png')
        plt.close()

        # Plot socket positions in 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(socket_positions[:, 0], socket_positions[:, 1], socket_positions[:, 2], c='r',
                   label='Socket Positions')
        ax.set_title(f'Socket Positions for {subassembly}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        plt.savefig(f'{output_folder}/socket_positions_3d.png')
        plt.close()

        # Plot plug positions in 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(plug_positions[:, 0], plug_positions[:, 1], plug_positions[:, 2], c='b', label='Plug Positions')
        ax.set_title(f'Plug Positions for {subassembly}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        plt.savefig(f'{output_folder}/plug_positions_3d.png')
        plt.close()

        # Calculate errors (distance between plug and socket positions)
        errors = np.linalg.norm(plug_positions - socket_positions, axis=1)

        # Plot error distribution
        plt.figure(figsize=(12, 8))
        plt.hist(errors, bins=20, edgecolor='k', alpha=0.7)
        plt.title(f'Error Distribution for {subassembly}')
        plt.xlabel('Error (Euclidean Distance)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'{output_folder}/error_distribution.png')
        plt.close()

        # Calculate and plot mean error
        mean_error = np.mean(errors)
        plt.figure(figsize=(12, 8))
        plt.bar(['Mean Error'], [mean_error], color='g')
        plt.title(f'Mean Error for {subassembly}')
        plt.ylabel('Error (Euclidean Distance)')
        plt.grid(True)
        plt.savefig(f'{output_folder}/mean_error.png')
        plt.close()


if __name__ == "__main__":
    # Example usage
    visualizer = GraspPoseVisualizer(num_envs=10)  # Adjust num_envs as necessary

    subassemblies = ['small_triangle', 'red_round_peg_1_5in', 'yellow_round_peg_2in', 'square_peg_hole_32mm_loose']

    for subassembly in subassemblies:
        visualizer._initialize_grasp_poses(subassembly, with_noise=True)
        visualizer.plot_and_save(subassembly)
