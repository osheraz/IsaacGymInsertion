# --------------------------------------------------------
# TODO
# https://arxiv.org/abs/todo
# Copyright (c) 2023 Osher Azulay
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


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
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

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        mlp_input_shape = input_shape[0]
        self.units = kwargs.pop('actor_units')
        out_size = self.units[-1]
        self.ft_info = kwargs.pop("ft_info")
        self.tactile_info = kwargs.pop("tactile_info")

        self.priv_mlp = kwargs.pop('priv_mlp_units')
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['extrin_adapt']

        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

            if self.priv_info_stage2:
                # ---- tactile Decoder ----
                # Dims of latent have to be the same |z_t - z'_t|
                if self.tactile_info:
                    path_checkpoint = None
                    if kwargs.pop('tactile_pretrained'):
                        # we should think about decoupling this problem and pretraining a decoder
                        import os
                        current_file = os.path.abspath(__file__)
                        self.root_dir = os.path.abspath(
                            os.path.join(current_file, "..", "..", "..", "..", "..")
                        )
                        path_checkpoint = kwargs.pop("checkpoint_tactile")

                    # load a simple tactile decoder
                    tactile_decoder_embed_dim = kwargs.pop('tactile_decoder_embed_dim')
                    self.tactile_decoder = load_tactile_resnet(tactile_decoder_embed_dim,
                                                               self.root_dir,
                                                               path_checkpoint,
                                                               pre_trained=kwargs.pop('tactile_pretrained'))

                    # add tactile mlp to the decoded features
                    self.tactile_units = kwargs.pop("tactile_units")
                    tactile_input_shape = kwargs.pop("tactile_input_shape")
                    self.tactile_mlp = MLP(
                        units=self.tactile_units, input_size=tactile_input_shape
                    )

                if self.ft_info:
                    assert 'ft is not supported yet, force rendering is currently ambiguous'
                    self.ft_units = kwargs.pop("ft_units")
                    ft_input_shape = kwargs.pop("ft_input_shape")
                    # self.ft_mlp = MLP(
                    #     units=self.ft_units, input_size=ft_input_shape[0]
                    # )
                    self.ft_adapt_tconv = FTAdaptTConv(ft_dim=ft_input_shape,
                                                       ft_out_dim=self.ft_units[-1])

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
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
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                # currently ft is useless.
                # extrin_ft = self.ft_adapt_tconv(obs_dict['ft_hist'])
                extrin_tactile = self._tactile_encode(obs_dict['tactile_hist'])
                # extrin = torch.cat([extrin_tactile, extrin_ft], dim=-1)

                extrin = extrin_tactile
                # during supervised training, extrin has gt label
                extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)  # Why tanh?
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin = torch.tanh(extrin) # layer normalize instead?
                obs = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
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
    def _tactile_encode(self, images):

        #                E, T,(finger) W, H, C  ->   E, T, C, W, H
        left_seq = images[:, :, 0, :, :, :].permute(0, 1, 4, 2, 3)
        right_seq = images[:, :, 1, :, :, :].permute(0, 1, 4, 2, 3)
        mid_seq = images[:, :, 2, :, :, :].permute(0, 1, 4, 2, 3)

        emb_left = self.tactile_decoder(left_seq).unsqueeze(1)
        emb_right = self.tactile_decoder(right_seq).unsqueeze(1)
        emb_bottom = self.tactile_decoder(mid_seq).unsqueeze(1)

        tactile_embeddings = torch.cat((emb_left, emb_right, emb_bottom), dim=-1)
        tac_emb = self.tactile_mlp(tactile_embeddings)

        return tac_emb


def load_tactile_resnet(tactile_decoder_embed_dim, root_dir, path_checkpoint, pre_trained=False):
    import algo.models.convnets.resnets as resnet
    import os

    tactile_decoder = resnet.resnet18(False, False, num_classes=tactile_decoder_embed_dim)

    if pre_trained:
        tactile_decoder.load_state_dict(os.path.join(root_dir, path_checkpoint))
        tactile_decoder.eval()
        for param in tactile_decoder.parameters():
            param.requires_grad = False
    return tactile_decoder
