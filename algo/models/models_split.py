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
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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


class ContactAE(nn.Module):
    def __init__(self, input_size, embedding_size=16):
        nn.Module.__init__(self)
        self.embedding_size = embedding_size
        self.contact_enc_mlp = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, embedding_size),
                                             nn.Tanh())
        self.contact_dec_mlp = nn.Sequential(nn.Linear(embedding_size, 32), nn.ReLU(), nn.Linear(32, input_size))

    def forward_enc(self, x):
        return self.contact_enc_mlp(x)

    def forward_dec(self, x):
        return self.contact_dec_mlp(x)


class ActorCriticSplit(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)

        actions_num = kwargs['actions_num']
        input_shape = kwargs['input_shape']
        mlp_input_shape = input_shape[0]
        self.units = kwargs['actor_units']
        self.contact_info = kwargs['gt_contacts_info']
        self.only_contact = kwargs['only_contact']

        self.priv_mlp_units = kwargs['priv_mlp_units']
        self.priv_info = kwargs['priv_info']
        self.priv_info_dim = kwargs['priv_info_dim']
        self.shared_parameters = kwargs['shared_parameters']
        self.vt_policy = kwargs['vt_policy']

        if self.priv_info:

            mlp_input_shape += self.priv_mlp_units[-1]
            self.env_mlp = MLP(units=self.priv_mlp_units, input_size=self.priv_info_dim)

            if self.contact_info:
                self.contact_mlp_units = kwargs['contacts_mlp_units']

                self.contact_ae = ContactAE(input_size=kwargs["num_contact_points"],
                                            embedding_size=self.contact_mlp_units[-1])
                if not self.only_contact:
                    mlp_input_shape += self.contact_mlp_units[-1]

        if self.vt_policy:
            student_cfg = kwargs['full_config']
            student_cfg.offline_train.model.use_tactile = False
            student_cfg.offline_train.model.use_seg = True
            student_cfg.offline_train.model.use_lin = True
            student_cfg.offline_train.model.use_img = True
            student_cfg.offline_train.only_bc = False

            student_cfg.offline_train.model.transformer.output_size = 32
            from algo.models.transformer.runner import Runner as Student

            self.stud_model = Student(student_cfg).model
            mlp_input_shape += 32

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        if not self.shared_parameters:
            self.critic_mlp = MLP(units=self.units, input_size=mlp_input_shape)

        self.value = layer_init(torch.nn.Linear(self.units[-1], 1), std=1.0)
        self.mu = layer_init(torch.nn.Linear(self.units[-1], actions_num), std=0.01)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

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
        mu, logstd, value, _, _ = self.actor_critic(obs_dict)
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
    def full_act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, latent_gt = self.actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),  # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'latent_gt': latent_gt,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, latent, latent_gt = self.actor_critic(obs_dict)
        latent = latent_gt if latent is None else latent
        return mu, latent

    def act_with_grad(self, obs_dict):
        # used for testing
        mu, logstd, value, latent, _ = self.actor_critic(obs_dict)
        return mu, latent

    def actor_critic(self, obs_dict, display=False):

        obs = obs_dict['obs']
        extrin, extrin_gt = None, None

        if self.priv_info and 'priv_info' in obs_dict:

            extrin_priv = self.env_mlp(obs_dict['priv_info'])

            if self.contact_info:
                extrin_contact = self.contact_ae.forward_enc(obs_dict['contacts'])
                if self.only_contact:
                    extrin_gt = extrin_contact
                else:
                    extrin_gt = torch.cat([extrin_priv, extrin_contact], dim=-1)
            else:
                extrin_gt = extrin_priv

        if 'latent' in obs_dict and obs_dict['latent'] is not None:

            extrin = obs_dict['latent']

            if display:
                plt.ylim(-1.2, 1.2)
                plt.scatter(list(range(extrin.shape[-1])), extrin.clone().detach().cpu().numpy()[0, :], color='r')
                plt.scatter(list(range(extrin_gt.shape[-1])), extrin_gt.clone().cpu().numpy()[0, :], color='b')
                plt.pause(0.0001)
                plt.cla()

            # predict with the student extrinsic
            obs = torch.cat([obs, extrin], dim=-1)
        elif self.priv_info:
            # predict with the teacher extrinsic
            obs = torch.cat([obs, extrin_gt], dim=-1)

        if self.vt_policy:

            img = obs_dict['img']
            seg = obs_dict['seg']
            student_obs = obs_dict['student_obs']

            if img.ndim == 3:
                img = img.reshape(*img.shape[:2], 1, 54, 96)
                seg = seg.reshape(*seg.shape[:2], 1, 54, 96)
            if img.ndim == 2:
                img = img.reshape(*img.shape[:1], 1,  1, 54, 96)
                seg = seg.reshape(*seg.shape[:1], 1,  1, 54, 96)

            valid_mask = ((seg == 2) | (seg == 3)).float()

            seg = seg * valid_mask
            img = img * valid_mask

            latent = self.stud_model(None, img, seg, student_obs)
            obs = torch.cat([obs, latent], dim=-1)

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
        mu, logstd, value, extrin, extrin_gt = self.actor_critic(input_dict)
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
