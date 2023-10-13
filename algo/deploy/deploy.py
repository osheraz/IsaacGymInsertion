# --------------------------------------------------------
# now its our turn.
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Co.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import numpy as np
import rospy

from algo.models.models import ActorCritic
from algo.models.running_mean_std import RunningMeanStd
import torch
import os
import hydra


class HardwarePlayer(object):
    def __init__(self, output_dir, full_config):

        self.action_scale = full_config.task.rl.pos_action_scale
        self.action_scale = full_config.task.rl.rot_action_scale

        self.device = full_config["rl_device"]
        # ------
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.env_config = full_config.task.env
        self.full_config = full_config
        # ---- build environment ----
        self.obs_shape = (self.env_config.numObservations,)
        self.num_actions = self.env_config.numActions
        self.num_targets = self.env_config.numTargets

        # ---- Tactile Info ---
        self.tactile_info = self.ppo_config.tactile_info
        self.tactile_seq_length = self.ppo_config.tactile_seq_length
        self.tactile_info_dim = self.network_config.tactile_mlp.units[0]
        # ---- ft Info --- TODO currently we dont use ft
        self.ft_info = self.ppo_config.ft_info
        self.ft_seq_length = self.ppo_config.ft_seq_length
        self.ft_input_dim = self.ppo_config.ft_input_dim
        self.ft_info_dim = self.ft_input_dim * self.ft_seq_length
        # ---- Priv Info ----
        self.priv_info = self.ppo_config.priv_info
        self.priv_info_dim = self.ppo_config.priv_info_dim
        self.extrin_adapt = self.ppo_config.extrin_adapt

        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.num_actions,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'input_shape': self.obs_shape,
            'extrin_adapt': True,
            'priv_info_dim': self.priv_info_dim,
            'priv_info': True,
            "tactile_info": self.tactile_info,
            "tactile_input_shape": self.tactile_info_dim,
            "ft_input_shape": self.ft_info_dim,
            "ft_info": self.ft_info,
            "tactile_units": self.network_config.tactile_mlp.units,
            "tactile_decoder_embed_dim": self.network_config.tactile_mlp.units[0],
            "ft_units": self.network_config.ft_mlp.units,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()

        # ---- Output Dir ----
        self.output_dir = output_dir
        self.dp_dir = os.path.join(self.output_dir, 'deploy')
        os.makedirs(self.dp_dir, exist_ok=True)

        tactile_info_path = '../allsight/experiments/conf/test.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_tactile = hydra.compose(config_name=tactile_info_path)['']['']['']['allsight']['experiments']['conf']

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory'][
            'yaml']  # strip superfluous nesting

    def _create_asset_info(self, i):

        subassembly = self.full_config.env.desired_subassemblies[i]
        components = list(self.asset_info_insertion[subassembly])
        rospy.logwarn('Parameters load for: {} --- >  {}'.format(components[0], components[1]))

        self.plug_height = self.asset_info_insertion[subassembly][components[0]]['length']
        self.socket_height = self.asset_info_insertion[subassembly][components[1]]['height']
        if any('rectangular' in sub for sub in components):
            self.plug_depth = self.asset_info_insertion[subassembly][components[0]]['width']
            self.plug_width = self.asset_info_insertion[subassembly][components[0]]['depth']
            self.socket_width = self.asset_info_insertion[subassembly][components[1]]['width']
            self.socket_depth = self.asset_info_insertion[subassembly][components[1]]['depth']
        else:
            self.plug_width = self.asset_info_insertion[subassembly][components[0]]['diameter']
            self.socket_width = self.asset_info_insertion[subassembly][components[1]]['diameter']

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.plug_grasp_pos_local = self.plug_height * 0.5 * torch.tensor([0.0, 0.0, 1.0],
                                                                          device=self.device).unsqueeze(0)
        self.plug_grasp_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.plug_tip_pos_local = self.plug_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)
        self.socket_tip_pos_local = self.socket_height * torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)

        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.targets = torch.zeros((1, self.env_config.numTargets), device=self.device)
        self.prev_targets = torch.zeros((1, self.env_config.numTargets), dtype=torch.float, device=self.device)

        # Keep track of history
        self.arm_joint_queue = torch.zeros((1, self.env_config.numObsHist, 7), dtype=torch.float, device=self.device)
        self.arm_vel_queue = torch.zeros((1, self.env_config.numObsHist, 7), dtype=torch.float, device=self.device)
        self.actions_queue = torch.zeros((1, self.env_config.numObsHist, self.num_actions),
                                         dtype=torch.float, device=self.device)
        self.targets_queue = torch.zeros((1, self.env_config.numObsHist, self.num_targets),
                                         dtype=torch.float, device=self.device)
        self.eef_queue = torch.zeros((1, self.env_config.numObsHist, 7),
                                     dtype=torch.float, device=self.device)
        self.goal_noisy_queue = torch.zeros((1, self.env_config.numObsHist, 7),
                                            dtype=torch.float, device=self.device)

        # tactile buffers
        self.tactile_imgs = torch.zeros(
            (1, 3,  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height, 3),
            device=self.device,
            dtype=torch.float,
        )
        # Way too big tensor.
        self.tactile_queue = torch.zeros(
            (1, self.tactile_seq_length, 3,  # left, right, bottom
             self.cfg_tactile.decoder.width, self.cfg_tactile.decoder.height, 3),
            device=self.device,
            dtype=torch.float,
        )

        self.ft_queue = torch.zeros((1, self.ft_seq_length, 6), device=self.device, dtype=torch.float)

    def deploy(self):
        from algo.deploy.env.env import ExperimentEnv
        # try to set up rospy
        rospy.init_node('DeployEnv')
        env = ExperimentEnv()
        rospy.logwarn('sda')
        # Wait for connections.
        rospy.sleep(0.5)

        hz = 60
        ros_rate = rospy.Rate(hz)

        # TODO command to the initial position
        env.move_to_init_state()

        obses = env.get_obs()
        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 16 * 3 * 2)).astype(np.float32)).cuda()
        proprio_hist_buf = torch.from_numpy(np.zeros((1, 30, 16 * 2)).astype(np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.env_dof_lower, self.env_dof_upper)[None]

        for i in range(3):
            obs_buf[:, i * 16 + 0:i * 16 + 16] = cur_obs_buf.clone()  # joint position
            obs_buf[:, i * 16 + 16:i * 16 + 32] = prev_target.clone()  # current target (obs_t-1 + s * act_t-1)

        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 16:32] = prev_target.clone()

        while True:
            obs = self.running_mean_std(obs_buf.clone())
            input_dict = {
                'obs': obs,
                'proprio_hist': self.sa_mean_std(proprio_hist_buf.clone()),
            }
            action = self.model.act_inference(input_dict)
            action = torch.clamp(action, -1.0, 1.0)

            target = prev_target + self.action_scale * action
            target = torch.clip(target, self.env_dof_lower, self.env_dof_upper)
            prev_target = target.clone()
            # interact with the hardware
            commands = target.cpu().numpy()[0]
            env.command_joint_position(commands)
            ros_rate.sleep()  # keep 20 Hz command
            # get o_{t+1}
            obses, torques = env.poll_joint_position(wait=True)
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            cur_obs_buf = unscale(obses, self.env_dof_lower, self.env_dof_upper)[None]
            prev_obs_buf = obs_buf[:, 32:].clone()
            obs_buf[:, :64] = prev_obs_buf
            obs_buf[:, 64:80] = cur_obs_buf.clone()
            obs_buf[:, 80:96] = target.clone()

            priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
            cur_proprio_buf = torch.cat([
                cur_obs_buf, target.clone()
            ], dim=-1)[:, None]
            proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
