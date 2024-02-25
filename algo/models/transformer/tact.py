from efficientnet_pytorch import EfficientNet
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
            self, obs_img: torch.tensor, obs_lin: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model
        Args:
            obs_img (torch.Tensor): batch of observations
            obs_lin (torch.Tensor): batch of lin observations

        Returns:
            extrinsic latent (torch.Tensor): predicted distance to goal
        """
        raise NotImplementedError


class TacT(BaseModel):
    def __init__(
            self,
            context_size: int = 3,
            num_channels: int = 3,
            num_lin_features: int = 10,
            num_outputs: int = 5,
            obs_encoder: Optional[str] = "efficientnet-b0",
            obs_encoding_size: Optional[int] = 512,
            lin_encoding_size: Optional[int] = 512,
            mha_num_attention_heads: Optional[int] = 2,
            mha_num_attention_layers: Optional[int] = 2,
            mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        Modified ViT class: uses a Transformer-based architecture to encode (current and past) visual observations
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
        """
        super(TacT, self).__init__(context_size, num_outputs)

        self.obs_encoding_size = obs_encoding_size
        self.num_lin_features = num_lin_features
        self.lin_encoding_size = lin_encoding_size
        self.num_channels = num_channels

        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=num_channels)  # context
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError

        self.lin_encoder = nn.Sequential(nn.Linear(num_lin_features, lin_encoding_size // 2),
                                         nn.ReLU(),
                                         nn.Linear(lin_encoding_size // 2, lin_encoding_size))

        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.latent_predictor = nn.Sequential(
            nn.Linear(32, self.num_output_params),
        )

    def forward(
            self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # currently, the size of lin_encoding is [batch_size, num_lin_features]
        lin_encoding = self.lin_encoder.extract_features(goal_img)
        if len(lin_encoding.shape) == 2:
            lin_encoding = lin_encoding.unsqueeze(1)
        # currently, the size of goal_encoding is [batch_size, 1, self.goal_encoding_size]
        assert lin_encoding.shape[2] == self.lin_encoding_size

        # split the observation into context based on the context size
        # image size is [batch_size, C*self.context_size, H, W]
        obs_img = torch.split(obs_img, self.num_channels, dim=1)

        # image size is [batch_size*self.context_size, self.num_channels, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # currently the size is [batch_size*(self.context_size + 1), 1280, H/32, W/32]
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        # currently the size is [batch_size*(self.context_size + 1), 1280, 1, 1]
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        # currently, the size is [batch_size, self.context_size+2, self.obs_encoding_size]

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, lin_encoding), dim=1)
        final_repr = self.decoder(tokens)
        # currently, the size is [batch_size, 32]

        latent_pred = self.latent_predictor(final_repr)

        return latent_pred
