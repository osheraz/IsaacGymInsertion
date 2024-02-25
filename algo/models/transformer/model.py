from torch import nn
import torch
import math
from algo.models.models import load_tactile_resnet


class TactileTransformer(nn.Module):
    def __init__(self, lin_input_size,
                 in_channels,
                 out_channels,
                 kernel_size,
                 embed_size,
                 hidden_size,
                 num_heads,
                 max_sequence_length,
                 num_layers,
                 output_size,
                 layer_norm=False):

        super(TactileTransformer, self).__init__()

        self.batch_first = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        # self.linear_in = nn.Linear(lin_input_size, 14)  # removed embed_size // 2 for no cnn
        self.linear_in = nn.Sequential(nn.Linear(lin_input_size, 64), nn.ReLU(), nn.Linear(64, 14))

        # for cnn_embedding, input is (B*T, C=1, W, H) and output is (B*T, 6)
        self.cnn_embedding = ConvEmbedding(in_channels, out_channels, kernel_size)
        # self.cnn_embedding = load_tactile_resnet(embed_size , num_channels=in_channels)

        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                        dim_feedforward=hidden_size, batch_first=self.batch_first)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.activation = nn.ReLU()

        self.linear_out = nn.Linear(embed_size, embed_size)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm_in = nn.LayerNorm(embed_size)
            self.layer_norm_out = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(0.2)
        # self.out = nn.Sequential(nn.Linear(embed_size, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, output_size))
        self.out = nn.Sequential(nn.Linear(embed_size, 16), nn.ReLU(), nn.Linear(16, output_size))

    def forward(self, cnn_input, lin_input, batch_size, embed_size, src_mask=None):

        lin_x = self.linear_in(lin_input)
        # process each finger seperate.
        cnn_embeddings = [self.cnn_embedding(cnn) for cnn in cnn_input]
        cnn_x = torch.cat(cnn_embeddings, dim=-1)
        cnn_x = cnn_x.view(batch_size, self.max_sequence_length, -1)
        x = torch.cat([lin_x, cnn_x], dim=-1)
        # x = lin_x
        # if self.layer_norm:
        #     x = self.layer_norm_in(x)
        # x = self.dropout(x)
        x = self.positonal_embedding(x)
        if src_mask is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, mask=src_mask)
        x = self.linear_out(x)
        # if self.layer_norm:
        #     x = self.layer_norm_out(x)
        # x = self.dropout(x)
        x = self.activation(x)
        x = self.out(x)
        # x = torch.tanh(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].permute(1, 0, 2)
        return self.dropout(x)


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvEmbedding, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(8, out_channels, kernel_size=kernel_size, stride=1)

        self.batchnorm1 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=3)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((3, 2))

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        # x = self.batchnorm1(x)
        x = self.activation(x)

        # x = self.maxpool1(x)
        # x = self.dropout1(x)
        x = self.conv2(x)

        # x = self.batchnorm2(x)
        x = self.activation(x)

        # x = self.max_pool2(x)
        # x = self.dropout2(x)
        x = self.global_avg_pool(x)

        x = x.flatten(start_dim=1)

        return x


# for tests
if __name__ == "__main__":
    lin_input_size = 33
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    embed_size = 32
    hidden_size = 32
    num_heads = 2
    max_sequence_length = 100
    num_layers = 2
    output_size = 8

    transformer = TactileTransformer(lin_input_size, in_channels, out_channels, kernel_size, embed_size, hidden_size,
                                     num_heads, max_sequence_length, num_layers, output_size)

    lin_x = torch.randn(2, 100, 33)
    cnn_x = [torch.randn(2 * 100, 1, 224, 224) for _ in range(3)]
    # cnn_x = cnn_x.view(2 * 100, 1, 224, 224)

    x = transformer(cnn_x, lin_x, 2, 16)
    print(x.shape)
