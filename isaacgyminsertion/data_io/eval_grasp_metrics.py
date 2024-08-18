import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist

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

        # Extract data
        socket_positions = self.init_socket_pos[subassembly].numpy()
        plug_positions = self.init_plug_pos[subassembly].numpy()
        num_poses = self.total_init_poses[subassembly]

        # Plot socket positions
        plt.figure(figsize=(12, 8))
        plt.scatter(socket_positions[:, 0], socket_positions[:, 1], c='r', label='Socket Positions')
        plt.title(f'Socket Positions for {subassembly}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_folder}/socket_positions.png')
        plt.close()

        # Plot plug positions
        plt.figure(figsize=(12, 8))
        plt.scatter(plug_positions[:, 0], plug_positions[:, 1], c='b', label='Plug Positions')
        plt.title(f'Plug Positions for {subassembly}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_folder}/plug_positions.png')
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

        # Additional metrics and plots can be added here

if __name__ == "__main__":
    # Example usage
    visualizer = GraspPoseVisualizer(num_envs=10)  # Adjust num_envs as necessary

    subassemblies = ['small_triangle', 'red_round_peg_1_5in', 'yellow_round_peg_2in', 'square_peg_hole_32mm_loose']

    for subassembly in subassemblies:
        visualizer._initialize_grasp_poses(subassembly, with_noise=True)
        visualizer.plot_and_save(subassembly)
