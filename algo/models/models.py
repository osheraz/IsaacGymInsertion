# --------------------------------------------------------
# TODO
# https://arxiv.org/abs/todo
# Copyright (c) 2024 Osher & friends?
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# --------------------------------------------------------
# based on: In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(layer_init(nn.Linear(input_size, output_size)))
            layers.append(nn.Tanh())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AdaptTConv(nn.Module):
    def __init__(self, ft_dim=6 * 5, ft_out_dim=32):
        super(AdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(ft_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, ft_out_dim)
        self.ft_dim = ft_dim
    def forward(self, tac, img, seg, x):

        x = x.reshape(x.shape[0], 30, self.ft_dim)
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)

        actions_num = kwargs['actions_num']
        input_shape = kwargs['input_shape']
        mlp_input_shape = input_shape[0]
        self.units = kwargs['actor_units']
        out_size = self.units[-1]
        self.ft_info = kwargs["ft_info"]
        self.tactile_info = kwargs["tactile_info"]
        self.obs_info = kwargs["obs_info"]
        self.contact_info = kwargs['gt_contacts_info']
        self.contact_mlp_units = kwargs['contacts_mlp_units']
        self.only_contact = kwargs['only_contact']
        self.priv_mlp_units = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['extrin_adapt']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.shared_parameters = kwargs['shared_parameters']

        self.temp_latent = []
        self.temp_extrin = []

        if self.priv_info:
            mlp_input_shape += self.priv_mlp_units[-1]

            if self.contact_info:
                self.priv_info_dim += self.contact_mlp_units[-1]
                self.contact_mlp = MLP(units=self.contact_mlp_units, input_size=kwargs["num_contact_points"])

            self.env_mlp = MLP(units=self.priv_mlp_units, input_size=self.priv_info_dim)

            if self.priv_info_stage2:
                # ---- tactile Decoder ----
                # Dims of latent have to be the same |z_t - z'_t|
                if self.obs_info:
                    self.obs_units = kwargs["obs_units"]
                    self.obs_mlp = MLP(
                        units=self.obs_units, input_size=kwargs["student_obs_input_shape"])

                if self.tactile_info:
                    path_checkpoint, root_dir = None, None
                    if False:  # kwargs['tactile_pretrained']
                        # we should think about decoupling this problem and pretraining a decoder
                        import os
                        current_file = os.path.abspath(__file__)
                        root_dir = os.path.abspath(
                            os.path.join(current_file, "..", "..", "..", "..", "..")
                        )
                        path_checkpoint = kwargs["checkpoint_tactile"]

                    # load a simple tactile decoder
                    tactile_decoder_embed_dim = kwargs['tactile_decoder_embed_dim']
                    tactile_input_dim = kwargs['tactile_input_dim']
                    num_channels = tactile_input_dim[-1]
                    num_fingers = 3

                    # self.tactile_decoder_m = load_tactile_resnet(tactile_decoder_embed_dim, 3 * num_channels)
                    self.tactile_decoder = load_tactile_resnet(tactile_decoder_embed_dim // num_fingers, num_channels)

                    # add tactile mlp to the decoded features
                    self.tactile_units = kwargs["mlp_tactile_units"]
                    if not self.obs_info and self.tactile_info:
                        self.tactile_units[-1] = self.priv_mlp_units[-1]  # self.tactile_units[-1] // 2
                    tactile_input_shape = kwargs["mlp_tactile_input_shape"]

                    self.tactile_mlp = MLP(
                        units=self.tactile_units, input_size=tactile_input_shape
                    )

                if self.obs_info and self.tactile_info:
                    self.merge_units = kwargs["merge_units"]
                    self.merge_mlp = MLP(
                        units=self.merge_units, input_size=self.tactile_units[-1] + self.obs_units[-1]
                    )

                if self.ft_info:
                    assert 'ft is not supported yet, force rendering is currently ambiguous'
                    self.ft_units = kwargs["ft_units"]
                    ft_input_shape = kwargs["ft_input_shape"]
                    self.ft_adapt_tconv = FTAdaptTConv(ft_dim=ft_input_shape,
                                                       ft_out_dim=self.ft_units[-1])

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        if not self.shared_parameters:
            self.critic_mlp = MLP(units=self.units, input_size=mlp_input_shape)

        self.value = layer_init(torch.nn.Linear(out_size, 1), std=1.0)
        self.mu = layer_init(torch.nn.Linear(out_size, actions_num), std=0.01)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        self.fig = plt.figure(figsize=(8, 6))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),  # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, latent, _ = self._actor_critic(obs_dict)
        return mu, latent

    def _actor_critic(self, obs_dict, display=False):

        obs = obs_dict['obs']
        extrin, extrin_gt = None, None

        # Transformer student ( latent pass is in frozen_ppo)
        if 'latent' in obs_dict and obs_dict['latent'] is not None:
            extrin = obs_dict['latent']
            obs = torch.cat([obs, extrin], dim=-1)

            if 'priv_info' in obs_dict:
                extrin_gt = self.env_mlp(obs_dict['priv_info'])
                # extrin_gt = torch.tanh(extrin_gt)

                if display:
                    plt.ylim(-1, 1)
                    plt.scatter(list(range(extrin_gt.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :],
                                color='r')
                    plt.scatter(list(range(extrin_gt.shape[-1])), extrin_gt.clone().cpu().numpy()[0, :], color='b')
                    plt.pause(0.0001)
                    plt.cla()

        # MLP models
        else:
            # Contact obs with extrin/gt_extrin and pass to the actor
            if self.priv_info:
                if self.priv_info_stage2:

                    if self.tactile_info:
                        extrin_tactile = self._tactile_encode_multi(obs_dict['tactile_hist'])
                    if self.obs_info:
                        extrin_obs = self.obs_mlp(obs_dict['student_obs'])
                    # If both, merge and create student extrin
                    if self.obs_info and self.tactile_info:
                        extrin = torch.cat([extrin_tactile, extrin_obs], dim=-1)
                        extrin = self.merge_mlp(extrin)
                    elif self.tactile_info:
                        extrin = extrin_tactile
                    else:
                        extrin = extrin_obs

                    # During supervised training, pass to priv_mlp -> extrin has gt label
                    with torch.no_grad():
                        if 'priv_info' in obs_dict:
                            if self.contact_info:
                                contact_features = self.contact_mlp(obs_dict['contacts'])
                                if self.only_contact:
                                    extrin_gt = contact_features
                                else:
                                    priv_obs = torch.cat([obs_dict['priv_info'], contact_features], dim=-1)
                                    extrin_gt = self.env_mlp(priv_obs)  # extrin
                            else:
                                extrin_gt = self.env_mlp(obs_dict['priv_info'])
                        else:
                            # In case we are evaluating stage2
                            extrin_gt = extrin

                    # extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                    # extrin_gt = torch.tanh(extrin_gt)
                    # extrin = torch.tanh(extrin)

                    # Applying action with student model
                    obs = torch.cat([obs, extrin], dim=-1)

                    # plot for latent viz
                    if display:
                        plt.ylim(-1, 1)
                        plt.scatter(list(range(extrin.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :],
                                    color='r')
                        plt.scatter(list(range(extrin_gt.shape[-1])), extrin_gt.clone().cpu().numpy()[0, :], color='b')
                        plt.pause(0.0001)
                        plt.cla()

                else:
                    # Stage1 -> Getting extrin from the priv_mlp
                    if self.contact_info:
                        contact_features = self.contact_mlp(obs_dict['contacts'])
                        if self.only_contact:
                            extrin = contact_features
                        else:
                            priv_obs = torch.cat([obs_dict['priv_info'], contact_features], dim=-1)
                            extrin = self.env_mlp(priv_obs)
                    else:
                        extrin = self.env_mlp(obs_dict['priv_info'])

                    # extrin = torch.tanh(extrin)

                    # plot for latent viz
                    if display and 'latent' in obs_dict:
                        plt.ylim(-1, 1)
                        plt.scatter(list(range(extrin.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :],
                                    color='b')
                        plt.scatter(list(range(extrin.shape[-1])), obs_dict['latent'].clone().cpu().numpy()[0, :],
                                    color='r')
                        plt.pause(0.0001)
                        plt.cla()

                    obs = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs)
        mu = self.mu(x)
        sigma = self.sigma
        if not self.shared_parameters:
            v = self.critic_mlp(obs)
            value = self.value(v)
        else:
            value = self.value(x)

        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result

    # @torch.no_grad()
    def _tactile_encode_multi(self, images):

        #                E, T,(finger) W, H, C  ->   E, T, C, W, H
        left_seq = images[:, :, 0, :, :, :].permute(0, 1, 4, 2, 3)
        right_seq = images[:, :, 1, :, :, :].permute(0, 1, 4, 2, 3)
        bot_seq = images[:, :, 2, :, :, :].permute(0, 1, 4, 2, 3)

        seq = torch.cat((left_seq, right_seq, bot_seq), dim=2)

        emb_multi = self.tactile_decoder_m(seq)

        tactile_embeddings = emb_multi

        tac_emb = self.tactile_mlp(tactile_embeddings)

        return tac_emb

    def _tactile_encode(self, images):

        #                E, T,(finger) W, H, C  ->   E, T, C, W, H
        left_seq = images[:, :, 0, :, :, :].permute(0, 1, 4, 2, 3)
        right_seq = images[:, :, 1, :, :, :].permute(0, 1, 4, 2, 3)
        bot_seq = images[:, :, 2, :, :, :].permute(0, 1, 4, 2, 3)

        emb_left = self.tactile_decoder(left_seq)
        emb_right = self.tactile_decoder(right_seq)
        emb_bottom = self.tactile_decoder(bot_seq)

        tactile_embeddings = torch.cat((emb_left, emb_right, emb_bottom), dim=-1)
        tac_emb = self.tactile_mlp(tactile_embeddings)

        return tac_emb


def load_tactile_resnet(embed_dim, num_channels,
                        root_dir=None, path_checkpoint=None, pre_trained=False):
    import algo.models.convnets.resnets as resnet
    import os

    tactile_encoder = resnet.resnet18(False, False, num_classes=embed_dim,
                                      num_channels=num_channels)

    if pre_trained:
        tactile_encoder.load_state_dict(os.path.join(root_dir, path_checkpoint))
        tactile_encoder.eval()
        for param in tactile_encoder.parameters():
            param.requires_grad = False

    return tactile_encoder
