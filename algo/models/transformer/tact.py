from efficientnet_pytorch import EfficientNet
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple, Callable
from algo.models.transformer.pointnets import PointNet
from algo.models.transformer.tactile_cnn import CNNWithSpatialSoftArgmax
# from algo.models.transformer.point_mae import MaskedPointNetEncoder

class DepthOnlyFCBackbone32x64(nn.Module):
    def __init__(self, latent_dim, output_activation=None, num_channel=3):
        super().__init__()

        self.num_channel = num_channel
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 32, 64]
            nn.Conv2d(in_channels=self.num_channel, out_channels=32, kernel_size=5),
            # [32, 28, 60]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 14, 30]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 12, 28]
            activation,
            nn.Flatten(),
            # Calculate the flattened size: 64 * 12 * 28 = 21,504
            nn.Linear(64 * 12 * 28, 128),
            activation,
            nn.Linear(128, latent_dim)

        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class DepthOnlyFCBackbone108x192(nn.Module):
    def __init__(self, latent_dim, output_activation=None, num_channel=1):
        super().__init__()

        self.num_channel = num_channel
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # Input: [1, 108, 192]
            nn.Conv2d(in_channels=self.num_channel, out_channels=32, kernel_size=5),
            # Output: [32, 104, 188]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: [32, 52, 94]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # Output: [64, 50, 92]
            activation,
            nn.Flatten(),
            # Calculate the flattened size: 64 * 50 * 92
            nn.Linear(64 * 50 * 92, 128),
            activation,
            nn.Linear(128, latent_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class DepthOnlyFCBackbone54x96(nn.Module):
    def __init__(self, latent_dim, output_activation=None, num_channel=1):
        super().__init__()

        self.num_channel = num_channel
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 54, 96]
            nn.Conv2d(in_channels=self.num_channel, out_channels=32, kernel_size=5),
            # [32, 50, 92]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 25, 46]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            # [64, 23, 44]
            activation,
            nn.Flatten(),
            # Calculate the flattened size: 64 * 23 * 44
            nn.Linear(64 * 23 * 44, 128),
            activation,
            nn.Linear(128, latent_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images)
        latent = self.output_activation(images_compressed)

        return latent

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                                   dim_feedforward=ff_dim_factor * embed_dim, activation="gelu",
                                                   batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len * embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x


class BaseModel(nn.Module):
    def __init__(
            self,
            context_size: int = 5,
            num_outputs: int = 5,
    ) -> None:
        """
        Base Model main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.num_output_params = num_outputs

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
            self, obs_tactile: torch.tensor, obs_lin: torch.tensor, contacts: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            obs_tactile (torch.Tensor): batch of observations
            obs_lin (torch.Tensor): batch of lin observations

        Returns:
            extrinsic latent (torch.Tensor): predicted distance to goal
        """
        raise NotImplementedError


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(MLPDecoder, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)

        return self.decoder(x)

class MultiModalModel(BaseModel):
    def __init__(
            self,
            context_size: int = 3,
            num_channels: int = 3,
            num_lin_features: int = 10,
            num_outputs: int = 5,
            share_encoding: Optional[bool] = True,
            stack_tactile: Optional[bool] = True,
            tactile_encoder: Optional[str] = "efficientnet-b0",
            img_encoder: Optional[str] = "efficientnet-b0",
            seg_encoder: Optional[str] = "efficientnet-b0",
            tactile_encoding_size: Optional[int] = 128,
            img_encoding_size: Optional[int] = 128,
            seg_encoding_size: Optional[int] = 128,
            lin_encoding_size: Optional[int] = 128,
            mha_num_attention_heads: Optional[int] = 2,
            mha_num_attention_layers: Optional[int] = 2,
            mha_ff_dim_factor: Optional[int] = 4,
            include_lin: Optional[bool] = True,
            include_img: Optional[bool] = True,
            include_seg: Optional[bool] = True,
            include_tactile: Optional[bool] = True,
            include_pcl: Optional[bool] = False,
            additional_lin: Optional[int] = 0,
            only_bc: Optional[bool] = False,
            pcl_conf: Optional[Dict] = None,
            use_transformer: Optional[bool] = True,
    ) -> None:
        """
        Modified ViT class: uses a Transformer-based architecture to encode (current and past) visual observations
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            tactile_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            tactile_encoding_size (int): size of the encoding of the observation images
        """
        super(MultiModalModel, self).__init__(context_size, num_outputs)

        self.tactile_encoding_size = tactile_encoding_size
        self.img_encoding_size = img_encoding_size
        self.seg_encoding_size = seg_encoding_size
        self.additional_lin = additional_lin
        self.alpha = 0.0

        if additional_lin:
            num_lin_features += additional_lin
        self.num_lin_features = num_lin_features
        self.num_channels = 3 if stack_tactile else 1
        self.share_encoding = share_encoding
        self.stack_tactile = stack_tactile

        self.include_lin = include_lin
        self.include_tactile = include_tactile
        self.include_img = include_img
        self.include_seg = include_img
        self.include_pcl = include_pcl

        self.tactile_encoder_type = tactile_encoder
        self.img_encoder_type = img_encoder
        self.seg_encoder_type = seg_encoder
        self.pcl_conf = pcl_conf

        num_features = 0

        if include_tactile:
            if tactile_encoder.split("-")[0] == "efficientnet":
                self.tactile_encoder = EfficientNet.from_name(tactile_encoder, in_channels=self.num_channels)
                self.tactile_encoder = replace_bn_with_gn(self.tactile_encoder)
                self.num_tactile_features = self.tactile_encoder._fc.in_features
            elif tactile_encoder == 'depth':
                self.tactile_encoder = CNNWithSpatialSoftArgmax(latent_dim=self.tactile_encoding_size)
                self.num_tactile_features = self.tactile_encoding_size
            else:
                raise NotImplementedError

            if self.num_tactile_features != self.tactile_encoding_size:
                self.compress_tac_enc = nn.Linear(self.num_tactile_features, self.tactile_encoding_size)
            else:
                self.compress_tac_enc = nn.Identity()

            tac_features = 1 if self.stack_tactile else 3
            num_features += tac_features

        if include_img:
            if img_encoder.split("-")[0] == "efficientnet":
                self.img_encoder = EfficientNet.from_name(img_encoder, in_channels=1)  # depth
                self.img_encoder = replace_bn_with_gn(self.img_encoder)
                self.num_img_features = self.img_encoder._fc.in_features
            elif img_encoder == 'depth':
                self.img_encoder = DepthOnlyFCBackbone54x96(latent_dim=self.img_encoding_size, num_channel=1)
                self.num_img_features = self.img_encoding_size
            else:
                raise NotImplementedError

            if self.num_img_features != self.img_encoding_size:
                self.compress_img_enc = nn.Linear(self.num_img_features, self.img_encoding_size)
            else:
                self.compress_img_enc = nn.Identity()

            num_features += 1

        if include_seg:
            if seg_encoder.split("-")[0] == "efficientnet":
                self.seg_encoder = EfficientNet.from_name(seg_encoder, in_channels=1)  # depth
                self.seg_encoder = replace_bn_with_gn(self.seg_encoder)
                self.num_seg_features = self.seg_encoder._fc.in_features
            elif seg_encoder == 'depth':
                self.seg_encoder = DepthOnlyFCBackbone54x96(latent_dim=self.seg_encoding_size, num_channel=1)
                self.num_seg_features = self.seg_encoding_size
            else:
                raise NotImplementedError

            if self.num_seg_features != self.seg_encoding_size:
                self.compress_seg_enc = nn.Linear(self.num_seg_features, self.seg_encoding_size)
            else:
                self.compress_seg_enc = nn.Identity()

            num_features += 1

        if include_lin:
            self.lin_encoding_size = lin_encoding_size
            self.lin_encoder = nn.Sequential(nn.Linear(num_lin_features, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, lin_encoding_size))

            num_features += 1

        if include_pcl:
            pcl_objects = 0
            self.pcl_encoder = nn.ModuleDict()

            if pcl_conf['merge_plug']:
                pcl_objects += 1
                # self.pcl_encoder['plug_encoder'] = MaskedPointNetEncoder(lin_encoding_size)
                self.pcl_encoder['plug_encoder'] = PointNet()

            if pcl_conf['merge_socket']:
                # self.pcl_encoder['socket_encoder'] = MaskedPointNetEncoder(lin_encoding_size)
                self.pcl_encoder['socket_encoder'] = PointNet()
                pcl_objects += 1

            if pcl_conf['merge_goal']:
                self.pcl_encoder['goal_encoder'] = PointNet()
                pcl_objects += 1

            if pcl_conf['scene_pcl']:
                self.pcl_encoder['scene_encoder'] = PointNet()
                pcl_objects += 1

            self.pcl_encoding_size = 256
            # self.compress_pcl_enc = nn.Linear(pcl_objects * self.pcl_encoding_size, self.lin_encoding_size)
            self.compress_pcl_enc = nn.Sequential(nn.Linear(pcl_objects * self.pcl_encoding_size, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, self.lin_encoding_size))
            num_features += 1

        if use_transformer and (self.context_size > 1 or include_tactile):
            self.decoder = MultiLayerDecoder(
                embed_dim=self.tactile_encoding_size,
                seq_len=self.context_size * num_features,
                output_layers=[256, 128, 64, 32],
                nhead=mha_num_attention_heads,
                num_layers=mha_num_attention_layers,
                ff_dim_factor=mha_ff_dim_factor,
            )

            # if include_tactile:
            #     self.new_decoder = MultiLayerDecoder(
            #         embed_dim=self.tactile_encoding_size,
            #         seq_len=self.context_size * (num_features+tac_features),
            #         output_layers=[256, 128, 64, 32],
            #         nhead=mha_num_attention_heads,
            #         num_layers=mha_num_attention_layers,
            #         ff_dim_factor=mha_ff_dim_factor,
            #     )
        else:
            self.decoder = MLPDecoder(
                input_dim=self.context_size * num_features * self.tactile_encoding_size,
                hidden_layers=[256, 128, 64],
                output_dim=32
            )
            # if include_tactile:
            #     self.new_decoder = MultiLayerDecoder(
            #         embed_dim=self.tactile_encoding_size,
            #         seq_len=self.context_size * (num_features+tac_features),
            #         output_layers=[256, 128, 64, 32],
            #         nhead=mha_num_attention_heads,
            #         num_layers=mha_num_attention_layers,
            #         ff_dim_factor=mha_ff_dim_factor,
            #     )

        self.latent_predictor = nn.Sequential(
            nn.Linear(32, self.num_output_params),
            nn.Tanh() if only_bc else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
            self, obs_tactile: torch.tensor, obs_img: torch.tensor, obs_seg: torch.tensor,
            lin_input: torch.tensor = None, obs_pcl: torch.tensor = None, add_lin_input: torch.tensor = None) -> torch.Tensor:

        tokens_list = []

        if self.include_tactile:
            # split the observation into context based on the context size
            B, T, F, C, W, H = obs_tactile.shape

            if self.stack_tactile:
                fingers = [obs_tactile.reshape(B*T, F*C, W, H)]
            else:
                fingers = [obs_tactile[:, :, i, ...].reshape(B*T, C, W, H) for i in range(F)]

            obs_features = []
            if self.share_encoding:
                if self.tactile_encoder_type.split("-")[0] == "efficientnet":
                    for finger in fingers:
                        # get the observation encoding
                        tactile_encoding = self.tactile_encoder.extract_features(finger)
                        # currently the size is [batch_size*(self.context_size), 1280, H/32, W/32]
                        tactile_encoding = self.tactile_encoder._avg_pooling(tactile_encoding)
                        # currently the size is [batch_size*(self.context_size), 1280, 1, 1]
                        if self.tactile_encoder._global_params.include_top:
                            tactile_encoding = tactile_encoding.flatten(start_dim=1)
                            tactile_encoding = self.tactile_encoder._dropout(tactile_encoding)
                        # currently, the size is [batch_size, self.context_size, self.tactile_encoding_size]
                        tactile_encoding = self.compress_tac_enc(tactile_encoding)
                        # currently, the size is [batch_size*(self.context_size), self.tactile_encoding_size]
                        # reshape the tactile_encoding to [context, batch, encoding_size], note that the order is flipped
                        tactile_encoding = tactile_encoding.reshape((self.context_size, -1, self.tactile_encoding_size))
                        tactile_encoding = torch.transpose(tactile_encoding, 0, 1)
                        obs_features.append(tactile_encoding)

                        # currently, the size is [batch_size, self.context_size, self.tactile_encoding_size]
                elif self.tactile_encoder_type == "depth":
                    for finger in fingers:  # in stack_tactile there is only 1
                        # get the observation encoding
                        tactile_encoding = self.tactile_encoder(finger)
                        tactile_encoding = tactile_encoding.reshape((self.context_size, -1, self.tactile_encoding_size))
                        tactile_encoding = torch.transpose(tactile_encoding, 0, 1)
                        obs_features.append(tactile_encoding)
            else:
                raise NotImplementedError

            obs_features = torch.cat(obs_features, dim=1)
            tokens_list.append(obs_features)

        if self.include_img:
            # img
            B, T, C, W, H = obs_img.shape
            obs_img = obs_img.reshape(B * T, C, W, H)
            if self.img_encoder_type.split("-")[0] == "efficientnet":
                img_encoding = self.img_encoder.extract_features(obs_img)
                # currently the size is [batch_size*(self.context_size), 1280, H/32, W/32]
                img_encoding = self.img_encoder._avg_pooling(img_encoding)
                # currently the size is [batch_size*(self.context_size), 1280, 1, 1]
                if self.img_encoder._global_params.include_top:
                    img_encoding = img_encoding.flatten(start_dim=1)
                    img_encoding = self.img_encoder._dropout(img_encoding)
                # currently, the size is [batch_size, self.context_size, self.img_encoding_size]
                img_encoding = self.compress_img_enc(img_encoding)
                # currently, the size is [batch_size*(self.context_size), self.img_encoding_size]
                # reshape the img_encoding to [context + 1, batch, encoding_size], note that the order is flipped
                img_encoding = img_encoding.reshape((self.context_size, -1, self.img_encoding_size))
                img_encoding = torch.transpose(img_encoding, 0, 1)

            elif self.img_encoder_type == "depth":
                # get the observation encoding
                img_encoding = self.img_encoder(obs_img)
                img_encoding = img_encoding.reshape((self.context_size, -1, self.img_encoding_size))
                img_encoding = torch.transpose(img_encoding, 0, 1)

            tokens_list.append(img_encoding)

        if self.include_seg:
            # img
            B, T, C, W, H = obs_seg.shape
            obs_seg = obs_seg.reshape(B * T, C, W, H)
            if self.seg_encoder_type.split("-")[0] == "efficientnet":
                seg_encoding = self.seg_encoder.extract_features(obs_seg)
                # currently the size is [batch_size*(self.context_size), 1280, H/32, W/32]
                seg_encoding = self.seg_encoder._avg_pooling(seg_encoding)
                # currently the size is [batch_size*(self.context_size), 1280, 1, 1]
                if self.seg_encoder._global_params.include_top:
                    seg_encoding = seg_encoding.flatten(start_dim=1)
                    seg_encoding = self.seg_encoder._dropout(seg_encoding)
                # currently, the size is [batch_size, self.context_size, self.seg_encoding_size]
                seg_encoding = self.compress_seg_enc(seg_encoding)
                # currently, the size is [batch_size*(self.context_size), self.seg_encoding_size]
                # reshape the seg_encoding to [context + 1, batch, encoding_size], note that the order is flipped
                seg_encoding = seg_encoding.reshape((self.context_size, -1, self.seg_encoding_size))
                seg_encoding = torch.transpose(seg_encoding, 0, 1)

            elif self.seg_encoder_type == "depth":
                # get the observation encoding
                seg_encoding = self.seg_encoder(obs_seg)
                seg_encoding = seg_encoding.reshape((self.context_size, -1, self.seg_encoding_size))
                seg_encoding = torch.transpose(seg_encoding, 0, 1)

            tokens_list.append(seg_encoding)

        # currently, the size of lin_encoding is [batch_size, num_lin_features]
        if self.include_lin:
            if self.additional_lin:
                assert NotImplementedError
                add_lin_input = add_lin_input
                lin_input = torch.cat((lin_input, add_lin_input), dim=2)

            if len(lin_input.shape) == 2:
                lin_input = lin_input.reshape((lin_input.shape[0], self.context_size, self.num_lin_features))

            lin_encoding = self.lin_encoder(lin_input)
            if len(lin_encoding.shape) == 2:
                lin_encoding = lin_encoding.unsqueeze(1)

            assert lin_encoding.shape[2] == self.lin_encoding_size

            tokens_list.append(lin_encoding)

        if self.include_pcl:
            pcl_encoding = []
            NP = 0
            if self.pcl_conf['merge_plug']:
                NP += self.pcl_conf['num_sample_plug']

                plug_pcl = obs_pcl[:, :NP].contiguous()
                pcl_encoding.append(self.pcl_encoder['plug_encoder'](plug_pcl))

            if self.pcl_conf['merge_socket']:
                NH = self.pcl_conf['num_sample_hole']
                socket_pcl = obs_pcl[:, NP: NP + NH].contiguous()
                pcl_encoding.append(self.pcl_encoder['socket_encoder'](socket_pcl))
                NP += NH

            if self.pcl_conf['merge_goal']:
                NG = self.pcl_conf['num_sample_goal']
                goal_pcl = obs_pcl[:, NP: NP + NG].contiguous()
                pcl_encoding.append(self.pcl_encoder['goal_encoder'](goal_pcl))
                NP += NG

            if self.pcl_conf['scene_pcl']:
                NA = self.pcl_conf['num_sample_all']
                all_pcl = obs_pcl[:, NP: NP + NA].contiguous()
                pcl_encoding.append(self.pcl_encoder['scene_encoder'](all_pcl))

            pcl_encoding = torch.cat(pcl_encoding, dim=-1)

            pcl_encoding = self.compress_pcl_enc(pcl_encoding)

            # pcl_encoding = self.pcl_encoder(obs_pcl)
            # obs_pcl = torch.cat([obs_pcl[:, :obs_pcl.shape[1] // 2],
            #                      obs_pcl[:, obs_pcl.shape[1] // 2:]], dim=-1)
            # obs_pcl = torch.cat([obs_pcl[:, :obs_pcl.shape[1] // 3],
            #                      obs_pcl[:, obs_pcl.shape[1] // 3: 2 * obs_pcl.shape[1] // 3],
            #                      obs_pcl[:, 2 * obs_pcl.shape[1] // 3:]
            #                      ], dim=-1)

            if len(pcl_encoding.shape) == 2:
                pcl_encoding = pcl_encoding.unsqueeze(1)

            tokens_list.append(pcl_encoding)

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat(tokens_list, dim=1)
        final_repr = self.decoder(tokens)

        # if self.include_tactile:
        #     tokens_list.append(obs_features)
        #     new_tokens = torch.cat(tokens_list, dim=1)
        #     new_repr = self.new_decoder(new_tokens)
        #     final_repr += self.alpha * new_repr

        # currently, the size is [batch_size, context, embed_dim]
        # currently, the size is [batch_size, 32]
        latent_pred = self.latent_predictor(final_repr)

        return latent_pred


# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module



def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module