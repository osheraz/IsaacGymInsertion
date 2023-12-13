# --------------------------------------------------------
# TODO
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher & Dhruv & friends?
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
    

class FTAdaptTConv(nn.Module):
    def __init__(self, ft_dim=6 * 5, ft_out_dim=32):
        super(FTAdaptTConv, self).__init__()
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

    def forward(self, x):
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
        self.priv_mlp_units = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['extrin_adapt']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.shared_parameters = kwargs['shared_parameters']


        self.temp_latent = []
        self.temp_extrin = []
        self.flag = True

        tactile_encoder_embed_dim = kwargs['tactile_encoder_embed_dim']
        tactile_input_dim = kwargs['tactile_input_dim']
        num_channels = tactile_input_dim[-1]
        num_fingers = 3
        self.tactile_encoder = load_tactile_resnet(tactile_encoder_embed_dim // num_fingers, num_channels)
        mlp_input_shape += 32 # (tactile_encoder_embed_dim // num_fingers) * num_fingers
        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        if not self.shared_parameters:
            self.critic_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        
        self.tactile_units = kwargs["mlp_tactile_units"]
        tactile_input_shape = kwargs["mlp_tactile_input_shape"]
        self.tactile_mlp = MLP(units=self.tactile_units, input_size=tactile_input_shape)
        self.contact_decoder = nn.Sequential(nn.Linear(self.tactile_units[-1], 32), nn.ReLU(), nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, kwargs['num_contact_points']))

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
        mu, logstd, value, _, _, dec = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),  # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'dec': dec,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, latent, _, dec = self._actor_critic(obs_dict)
        return mu, latent, dec

    def _actor_critic(self, obs_dict, display=False):
        
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
    
        tactile_emb = self._tactile_encode(obs_dict['tactile_hist'])
        contact_dec = self.contact_decoder(tactile_emb)
        obs = torch.cat([obs, tactile_emb], dim=-1)

        x = self.actor_mlp(obs)
        mu = self.mu(x)
        sigma = self.sigma
        if not self.shared_parameters:
            v = self.critic_mlp(obs)
            value = self.value(v)
        else:
            value = self.value(x)

        return mu, mu * 0 + sigma, value, extrin, extrin_gt, contact_dec

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value, extrin, extrin_gt, dec = self._actor_critic(input_dict)
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
            'dec': dec,
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

        emb_left = self.tactile_encoder(left_seq)
        emb_right = self.tactile_encoder(right_seq)
        emb_bottom = self.tactile_encoder(bot_seq)

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
