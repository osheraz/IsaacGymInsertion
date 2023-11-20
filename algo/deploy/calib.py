##### !/usr/bin/env /home/osher/Desktop/isaacgym/venv/bin/python3
# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import isaacgym
import rospy
import torch
from isaacgyminsertion.utils import torch_jit_utils
import isaacgyminsertion.tasks.factory_tactile.factory_control as fc
from time import time
import numpy as np
from hyperopt import fmin, hp, tpe, space_eval
import matplotlib.pyplot as plt


# import matplotlib
# matplotlib.use('TkAgg')
# import pyformulas as pf

class HardwarePlayer():

    def __init__(self, ):

        self.pos_scale_deploy = [0.0003, 0.0003, 0.0003]
        self.rot_scale_deploy = [0.0002, 0.0002, 0.001]
        self.device = 'cuda:0'

        self._initialize_trajectories()
        self._initialize_grasp_poses()
        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv')

        self.env = ExperimentEnv(with_hand=False, with_tactile=False)
        rospy.logwarn('Finished setting the env, lets play.')

    def _initialize_trajectories(self, gp='test_files/sim_traj/'):

        from glob import glob
        abs_path = '/home/robotics/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion'
        all_paths = glob(f'{abs_path}/{gp}/*.npz')
        print('Total trajectories:', len(all_paths))
        from isaacgyminsertion.utils.torch_jit_utils import matrix_to_quaternion

        self.arm_joints_sim = np.zeros((len(all_paths), 1999, 7))
        self.eef_pose_sim = np.zeros((len(all_paths), 1999, 7))
        lz = 1000
        self.actions_sim = np.zeros((len(all_paths), 1999, 6))
        # self.dones = np.zeros((len(all_paths), 1))

        def convert_quat_wxyz_to_xyzw(q):
            q[3], q[0], q[1], q[2] = q[0], q[1], q[2], q[3]
            return q

        for i, p in enumerate(all_paths):
            # data = np.load(p)
            data = np.load(p, allow_pickle=True)['arr_0'][()]  # saved without keys :X
            # done_idx = data['done'].nonzero()[-1][0]

            self.arm_joints_sim[i] = data['sim_joints']
            # to_matrix = data['sim_pose'][:, 3:].reshape(data['sim_pose'][:, 3:].shape[0], 3, 3)
            # quat = matrix_to_quaternion(torch.tensor(to_matrix)).numpy()
            # for j in range(len(quat)):f
            #     quat[j] = convert_quat_wxyz_to_xyzw(quat[j])

            self.eef_pose_sim[i][:, :] = data['sim_pose']#[:, :3]
            # self.eef_pose_sim[i][:, 3:] = quat

            self.actions_sim[i] = data['sim_actions']
            # self.dones[i] = done_idx

        lz = 1000
        self.arm_joints_sim = self.arm_joints_sim[:,:lz, :]
        self.eef_pose_sim = self.eef_pose_sim[:,:lz,:]
        self.actions_sim = self.actions_sim[:,:lz,:]
        #
        # for i in range(len(self.arm_joints_sim)):
        #     display = True
        #
        #     if display:
        #         ax1 = plt.subplot(7, 1, 1)
        #         ax2 = plt.subplot(7, 1, 2)
        #         ax3 = plt.subplot(7, 1, 3)
        #         ax4 = plt.subplot(7, 1, 4)
        #         ax5 = plt.subplot(7, 1, 5)
        #         ax6 = plt.subplot(7, 1, 6)
        #         ax7 = plt.subplot(7, 1, 7)
        #
        #         ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        #         traj = self.arm_joints_sim[i]
        #         for j in range(len(ax)):
        #             ax[j].plot(np.array(traj)[:, j], color='r', label='real')
        #             # ax[j].plot(sim_joints[:len(traj)][:, j], color='b', label='sim')
        #
        #         plt.legend()
        #         # plt.title(f"Total Error: {loss2} \n")
        #         plt.show()

        # well...

    def apply_action(self, actions, pose=None, do_scale=True, do_clamp=False, wait=True):

        actions = torch.tensor(actions, device=self.device, dtype=torch.float).unsqueeze(0)

        if pose is None:
            pos, quat = self.env.arm.get_ee_pose()
        else:
            pos, quat = pose[:3], pose[3:]

        self.fingertip_centered_pos = torch.tensor(pos, device=self.device, dtype=torch.float).unsqueeze(0)
        self.fingertip_centered_quat = torch.tensor(quat, device=self.device, dtype=torch.float).unsqueeze(0)

        # Apply the action
        if do_clamp:
            actions = torch.clamp(actions, -1.0, 1.0)
        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.pos_scale_deploy, device=self.device))
        self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.rot_scale_deploy, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_jit_utils.quat_from_angle_axis_deploy(angle, axis)

        clamp_rot_thresh = 1.0e-6
        rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > clamp_rot_thresh,
                                       rot_actions_quat,
                                       torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device))

        self.ctrl_target_fingertip_centered_quat = torch_jit_utils.quat_mul_deploy(rot_actions_quat,
                                                                                   self.fingertip_centered_quat)

        self.generate_ctrl_signals(wait=wait)

    def _initialize_grasp_poses(self, gp='yellow_round_peg_2in'):

        pp = '/home/robotics/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/'
        self.initial_grasp_poses = np.load(f'{pp}initial_grasp_data/{gp}.npz')

        self.total_init_poses = self.initial_grasp_poses['socket_pos'].shape[0]
        self.init_dof_pos = torch.zeros((self.total_init_poses, 15))
        self.init_dof_pos = self.init_dof_pos[:, :7]
        dof_pos = self.initial_grasp_poses['dof_pos'][:, :7]

        from tqdm import tqdm
        print("Loading init grasping poses for:", gp)
        for i in tqdm(range(self.total_init_poses)):
            self.init_dof_pos[i] = torch.from_numpy(dof_pos[i])

    def generate_ctrl_signals(self, wait=True):

        ctrl_info = self.env.get_info_for_control()

        fingertip_centered_jacobian_tf = torch.tensor(ctrl_info['jacob'], device=self.device).unsqueeze(0)

        arm_dof_pos = torch.tensor(ctrl_info['joints'], device=self.device).unsqueeze(0)

        cfg_ctrl = {'num_envs': 1,
                    'jacobian_type': 'geometric',
                    'ik_method': 'dls'
                    }

        self.ctrl_target_dof_pos = fc.compute_dof_pos_target_deploy(
            cfg_ctrl=cfg_ctrl,
            arm_dof_pos=arm_dof_pos,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            jacobian=fingertip_centered_jacobian_tf,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_gripper_dof_pos=0,
            device=self.device)

        self.ctrl_target_dof_pos = self.ctrl_target_dof_pos[:, :7]
        target_joints = self.ctrl_target_dof_pos.cpu().detach().numpy().squeeze().tolist()

        try:
            self.env.move_to_joint_values(target_joints, wait=wait)

        except:
            print(f'failed to reach {target_joints}')



def hyper_param_tune(hw):
    """
    Find best simulated PID values to fit real PID values
    """

    # Wait for connections.
    rospy.sleep(0.5)

    hz = 100
    ros_rate = rospy.Rate(hz)

    # hw.env.move_to_init_state()
    # hw.env.regularize_force(True)
    hw.env.arm.move_to_init()
    def objective(params):
        """
        objective function to be minimized
        :param params: PID values
        :return: loss

        """

        print(f"Current params:", params)

        # hw.pos_scale_deploy[0] = params['pos_scale_x']
        # hw.pos_scale_deploy[1] = params['pos_scale_y']
        # hw.pos_scale_deploy[2] = params['pos_scale_z']
        # hw.rot_scale_deploy[0] = params['rot_scale_r']
        # hw.rot_scale_deploy[1] = params['rot_scale_p']
        # hw.rot_scale_deploy[2] = params['rot_scale_y']

        # hw.env.arm.move_to_init()
        # hw.env.hand.set_gripper_joints_to_init()

        # Sample trajectory from the sim
        idx = np.random.randint(0, len(hw.arm_joints_sim))
        # done = int(hw.dones[idx][0])
        sim_joints = hw.arm_joints_sim[idx][:, :]
        sim_actions = hw.actions_sim[idx][:, :]
        sim_pose = hw.eef_pose_sim[idx][:, :]

        # Move to start of sim traj
        hw.env.move_to_joint_values(sim_joints[0], wait=True)

        # Pick and Place
        # hw.env.move_to_init_state()
        # grasp_joints_bit = [0.33339, 0.52470, 0.12685, -1.6501, -0.07662, 0.97147, -1.0839]
        # grasp_joints = [0.3347, 0.54166, 0.12498, -1.6596, -0.07943, 0.94501, -1.0817]
        # hw.env.move_to_joint_values(grasp_joints_bit, wait=True)
        # hw.env.move_to_joint_values(grasp_joints, wait=True)
        # hw.env.grasp()
        # hw.env.move_to_joint_values(grasp_joints_bit, wait=True)
        # hw.env.move_to_joint_values(hw.env.joints_above_plug, wait=True)
        # hw.env.move_to_joint_values(hw.env.joints_above_socket, wait=True)

        # Sample init error
        # random_init_idx = torch.randint(0, hw.total_init_poses, size=(1,))
        # kuka_dof_pos = hw.init_dof_pos[random_init_idx]
        # kuka_dof_pos = kuka_dof_pos.cpu().detach().numpy().squeeze().tolist()
        # hw.env.move_to_joint_values(kuka_dof_pos, wait=True)

        # Squeeze Squeeze
        # hw.env.grasp()
        # hw.env.grasp()

        rospy.sleep(1.0)

        traj = []
        pose = []
        actions = []

        # import random
        # def generate_random_list(size=6):
        #     return [random.choice([-1, 0, 1]) for _ in range(size)]
        #
        # action_list = generate_random_list()
        # action_list[2] = -1
        # action_list += [0, 0, 0]

        # print(action_list)

        # action_list = [1, 1, -1, 0, 0, 0]

        hw.env.arm.calib_robotiq()
        rospy.sleep(2.0)
        hw.env.arm.calib_robotiq()

        skip = 4
        for i in range(1, skip * len(sim_joints)):  # len(sim_joints)

            start_time = time()

            joints = hw.env.arm.get_joint_values()
            pos, quat = hw.env.arm.get_ee_pose()

            # eef_pos = sim_pose[i]
            action = sim_actions[0]
            # action_list = np.array(action_list)

            print(action)
            # Regularize Ft
            action = np.where(np.abs(hw.env.get_ft()) > 1.0, action * 0, action).tolist()

            # action = action_list
            #
            # if np.sign(quat[0]) != np.sign(eef_pos[3]):
            #     quat[0] *= -1
            #     quat[1] *= -1
            #     if np.sign(quat[0]) != np.sign(eef_pos[3]):
            #         print('check')

            pose.append(pos + quat)
            traj.append(joints)

            actions.append(action)
            hw.apply_action(actions=action, wait=False)  # pose=eef_pos,

            ros_rate.sleep()

        rospy.sleep(0.5)

        traj = np.array(traj)
        pose = np.array(pose)

        sim_joints = sim_joints[:-1, :]
        sim_pose = sim_pose[:-1, :]

        save = False
        if save:
            import os
            from datetime import datetime
            data_path = '/home/robotics/Downloads/osher'

            dict_to_save = {'real_joints': traj,
                            'real_pose': pose,
                            'real_actions': actions
                            }

            np.savez_compressed(os.path.join(data_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz'),
                                dict_to_save)
        # Loss function
        loss1 = 1  # np.sum((traj - sim_joints) ** 2)
        loss2 = 1  # np.sum((pose[:, :3] - sim_pose[:, :3]) ** 2)

        display = True
        if display:
            ax1 = plt.subplot(7, 1, 1)
            ax2 = plt.subplot(7, 1, 2)
            ax3 = plt.subplot(7, 1, 3)
            ax4 = plt.subplot(7, 1, 4)
            ax5 = plt.subplot(7, 1, 5)
            ax6 = plt.subplot(7, 1, 6)
            ax7 = plt.subplot(7, 1, 7)

            ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            for j in range(len(ax)):
                ax[j].plot(np.array(traj)[::skip, j], color='r', label='real', marker='o', markersize=0.2)
                ax[j].plot(sim_joints[:len(traj)][:, j], color='b', marker='o', label='sim', markersize=0.2)

            plt.legend()
            plt.title(f"Total Error: {loss2} \n")
            plt.show()

        print(f"Total Error: {loss2} \n")

        return loss2

    # Hyperparams space
    # Todo: we can optimize each scale individually if they are not correlated. it is faster.

    space = {
        "pos_scale_x": hp.uniform("pos_scale_x", 0.0, 0.01),
        "pos_scale_y": hp.uniform("pos_scale_y", 0.0, 0.01),
        "pos_scale_z": hp.uniform("pos_scale_z", 0.0, 0.01),
        "rot_scale_r": hp.uniform("rot_scale_r", 0.0, 0.001),
        "rot_scale_p": hp.uniform("rot_scale_p", 0.0, 0.001),
        "rot_scale_y": hp.uniform("rot_scale_y", 0.0, 0.01),
    }

    # TPE algo based on bayesian optimization
    algo = tpe.suggest
    # spark_trials = SparkTrials()
    best_result = fmin(
        fn=objective,
        space=space,
        algo=algo,
        max_evals=50)

    print(f"Best params: \n")
    print(space_eval(space, best_result))

    import yaml
    with open('./best_params.yaml', 'w') as outfile:
        yaml.dump(space_eval(space, best_result), outfile, default_flow_style=False)

    print("Finished")


if __name__ == '__main__':
    hw = HardwarePlayer()

    hyper_param_tune(hw)
