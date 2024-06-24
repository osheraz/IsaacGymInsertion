from efficientnet_pytorch import EfficientNet
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple, Callable


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


class MultiModalModel(BaseModel):
    def __init__(
            self,
            context_size: int = 3,
            num_channels: int = 3,
            num_lin_features: int = 10,
            num_outputs: int = 5,
            share_encoding: Optional[bool] = True,
            tactile_encoder: Optional[str] = "efficientnet-b0",
            img_encoder: Optional[str] = "efficientnet-b0",
            tactile_encoding_size: Optional[int] = 128,
            img_encoding_size: Optional[int] = 128,
            lin_encoding_size: Optional[int] = 128,
            mha_num_attention_heads: Optional[int] = 2,
            mha_num_attention_layers: Optional[int] = 2,
            mha_ff_dim_factor: Optional[int] = 4,
            include_lin: Optional[bool] = True,
            include_img: Optional[bool] = True,
            include_tactile: Optional[bool] = True,
            additional_lin: Optional[int] = 0,
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
        if additional_lin:
            num_lin_features += additional_lin
            self.additional_lin = additional_lin
        self.num_lin_features = num_lin_features
        self.num_channels = num_channels
        self.share_encoding = share_encoding

        self.include_lin = include_lin
        self.include_tactile = include_tactile
        self.include_img = include_img
        num_features = 0

        if include_tactile:
            if tactile_encoder.split("-")[0] == "efficientnet":
                self.tactile_encoder = EfficientNet.from_name(tactile_encoder, in_channels=num_channels)
                self.tactile_encoder = replace_bn_with_gn(self.tactile_encoder)
                self.num_tactile_features = self.tactile_encoder._fc.in_features
            else:
                raise NotImplementedError

            if self.num_tactile_features != self.tactile_encoding_size:
                self.compress_obs_enc = nn.Linear(self.num_tactile_features, self.tactile_encoding_size)
            else:
                self.compress_obs_enc = nn.Identity()

            num_features += 3

        if include_img:
            if img_encoder.split("-")[0] == "efficientnet":
                self.img_encoder = EfficientNet.from_name(img_encoder, in_channels=1)  # depth
                self.img_encoder = replace_bn_with_gn(self.img_encoder)
                self.num_img_features = self.img_encoder._fc.in_features
            else:
                raise NotImplementedError

            if self.num_img_features != self.img_encoding_size:
                self.compress_obs_enc = nn.Linear(self.num_img_features, self.img_encoding_size)
            else:
                self.compress_obs_enc = nn.Identity()

            num_features += 1

        if include_lin:
            self.lin_encoding_size = lin_encoding_size
            self.lin_encoder = nn.Sequential(nn.Linear(num_lin_features, lin_encoding_size // 2),
                                             nn.ReLU(),
                                             nn.Linear(lin_encoding_size // 2, lin_encoding_size))

            num_features += 1

        self.decoder = MultiLayerDecoder(
            embed_dim=self.tactile_encoding_size,
            seq_len=self.context_size * num_features,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.latent_predictor = nn.Sequential(
            nn.Linear(32, self.num_output_params),
        )

    def forward(
            self, obs_tactile: torch.tensor, obs_img: torch.tensor,
            lin_input: torch.tensor = None, add_lin_input: torch.tensor = None) -> torch.Tensor:

        tokens_list = []

        if self.include_tactile:
            # split the observation into context based on the context size
            B, T, F, C, W, H = obs_tactile.shape
            # obs_tactile = obs_tactile.reshape(B*T, CF, W, H)
            fingers = [obs_tactile[:, :, i, ...].reshape(B*T, C, W, H) for i in range(F)]

            obs_features = []
            if self.share_encoding:
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
                    tactile_encoding = self.compress_obs_enc(tactile_encoding)
                    # currently, the size is [batch_size*(self.context_size), self.tactile_encoding_size]
                    # reshape the tactile_encoding to [context + 1, batch, encoding_size], note that the order is flipped
                    tactile_encoding = tactile_encoding.reshape((self.context_size, -1, self.tactile_encoding_size))
                    tactile_encoding = torch.transpose(tactile_encoding, 0, 1)
                    # currently, the size is [batch_size, self.context_size, self.tactile_encoding_size]
                    obs_features.append(tactile_encoding)
            else:
                raise NotImplementedError

            obs_features = torch.cat(obs_features, dim=1)
            tokens_list.append(obs_features)

        if self.include_img:
            # img
            B, T, C, W, H = obs_img.shape
            obs_img = obs_img.reshape(B*T, C, W, H)
            img_encoding = self.img_encoder.extract_features(obs_img)
            # currently the size is [batch_size*(self.context_size), 1280, H/32, W/32]
            img_encoding = self.img_encoder._avg_pooling(img_encoding)
            # currently the size is [batch_size*(self.context_size), 1280, 1, 1]
            if self.img_encoder._global_params.include_top:
                img_encoding = img_encoding.flatten(start_dim=1)
                img_encoding = self.img_encoder._dropout(img_encoding)
            # currently, the size is [batch_size, self.context_size, self.img_encoding_size]
            img_encoding = self.compress_obs_enc(img_encoding)
            # currently, the size is [batch_size*(self.context_size), self.img_encoding_size]
            # reshape the img_encoding to [context + 1, batch, encoding_size], note that the order is flipped
            img_encoding = img_encoding.reshape((self.context_size, -1, self.img_encoding_size))
            img_encoding = torch.transpose(img_encoding, 0, 1)
            tokens_list.append(img_encoding)

        # currently, the size of lin_encoding is [batch_size, num_lin_features]
        if self.include_lin:
            if self.additional_lin:
                add_lin_input = add_lin_input
                lin_input = torch.cat((lin_input, add_lin_input), dim=2)
            lin_encoding = self.lin_encoder(lin_input)
            if len(lin_encoding.shape) == 2:
                lin_encoding = lin_encoding.unsqueeze(1)
            # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
            assert lin_encoding.shape[2] == self.lin_encoding_size
            tokens_list.append(lin_encoding)

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat(tokens_list, dim=1)

        final_repr = self.decoder(tokens)
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