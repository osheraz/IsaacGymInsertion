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

        self.pos_scale_deploy = [0.0015, 0.0015, 0.0015]
        self.rot_scale_deploy = [0.001, 0.001, 0.001]
        self.device = 'cuda:0'

        self._initialize_trajectories()

        from algo.deploy.env.env import ExperimentEnv
        rospy.init_node('DeployEnv')

        self.env = ExperimentEnv(with_hand=False, with_tactile=False)
        rospy.logwarn('Finished setting the env, lets play.')

    def _initialize_trajectories(self, gp='/home/robotics/Downloads'):

        from glob import glob

        all_paths = glob(f'{gp}/datastore_42_test/*/*.npz')
        print('Total trajectories:', len(all_paths))
        from isaacgyminsertion.utils.torch_jit_utils import matrix_to_quaternion

        self.arm_joints = np.zeros((len(all_paths), 500, 7))
        self.eef_pose_mat = np.zeros((len(all_paths), 500, 12))
        self.eef_pose = np.zeros((len(all_paths), 500, 7))

        self.actions = np.zeros((len(all_paths), 500, 6))
        self.dones = np.zeros((len(all_paths), 1))

        def convert_quat_wxyz_to_xyzw(q):
            q[3], q[0], q[1], q[2] = q[0], q[1], q[2], q[3]
            return q

        for i, p in enumerate(all_paths):
            data = np.load(p)
            done_idx = data['done'].nonzero()[-1][0]

            self.arm_joints[i] = data['arm_joints']
            self.eef_pose_mat[i] = data['eef_pos']

            to_matrix = self.eef_pose_mat[i][:, 3:].reshape(self.eef_pose_mat[i][:, 3:].shape[0], 3, 3)
            quat = matrix_to_quaternion(torch.tensor(to_matrix)).numpy()
            for j in range(len(quat)):
                quat[j] = convert_quat_wxyz_to_xyzw(quat[j])

            self.eef_pose[i][:, :3] = data['eef_pos'][:, :3]
            self.eef_pose[i][:, 3:] = quat

            self.actions[i] = data['action']
            self.dones[i] = done_idx

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

        hw.env.arm.move_to_init()

        idx = np.random.randint(0, len(hw.arm_joints))
        idx = 1

        done = int(hw.dones[idx][0])
        sim_joints = hw.arm_joints[idx][:done, :]
        sim_actions = hw.actions[idx][:done, :]
        sim_pose = hw.eef_pose[idx][:done, :]

        # move to start of sim traj
        hw.env.move_to_joint_values(sim_joints[0], wait=True)
        rospy.sleep(1.0)

        traj = []
        pose = []
        actions = []

        for i in range(1, len(sim_joints)):

            start_time = time()

            joints = hw.env.arm.get_joint_values()
            traj.append(joints)
            pos, quat = hw.env.arm.get_ee_pose()

            eef_pos = sim_pose[i]
            action = sim_actions[i]

            action = [0,0,0,0,0,0]

            if np.sign(quat[0]) != np.sign(eef_pos[3]):
                quat[0] *= -1
                quat[1] *= -1
                quat[3] *= -1
                if np.sign(quat[0]) != np.sign(eef_pos[3]):
                    print('check')

            pose.append(pos + quat)
            actions.append(action)
            hw.apply_action(actions=action, wait=False) # pose=eef_pos,

            ros_rate.sleep()

        rospy.sleep(0.5)

        traj = np.array(traj)
        pose = np.array(pose)

        sim_joints = sim_joints[:-1, :]
        sim_pose = sim_pose[:-1, :]

        save = True
        if save:
            import os
            from datetime import datetime
            data_path = ''

            dict_to_save = {'real_joints': traj,
                            'real_pose': pose,
                            'real_actions': actions
                            }

            np.savez_compressed(os.path.join(data_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz'),
                                dict_to_save)
        # Loss function
        loss1 = np.sum((traj - sim_joints) ** 2)
        loss2 = np.sum((pose[:, :3] - sim_pose[:, :3]) ** 2)

        display = False
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
                ax[j].plot(np.array(pose)[:, j], color='r', label='real')
                ax[j].plot(sim_pose[:len(pose)][:, j], color='b', label='sim')

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
        max_evals=500)

    print(f"Best params: \n")
    print(space_eval(space, best_result))

    import yaml
    with open('./best_params.yaml', 'w') as outfile:
        yaml.dump(space_eval(space, best_result), outfile, default_flow_style=False)

    print("Finished")


if __name__ == '__main__':
    hw = HardwarePlayer()

    hyper_param_tune(hw)
