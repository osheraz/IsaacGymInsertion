from torch import nn
import torch
import math


class TactileTransformer(nn.Module):
    def __init__(self, lin_input_size, in_channels, out_channels, kernel_size, embed_size, hidden_size, num_heads, max_sequence_length, num_layers, output_size, layer_norm=True):
        super(TactileTransformer, self).__init__()

        self.batch_first = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        self.linear_in = nn.Linear(lin_input_size, embed_size//2)

        self.cnn_embedding = ConvEmbedding(in_channels, out_channels, kernel_size)
        
        self.positonal_embedding = PositionalEncoding(embed_size, max_len=max_sequence_length)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=self.batch_first)
        
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
        self.activation = nn.ELU()

        self.linear_out = nn.Linear(embed_size, embed_size)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm_in = nn.LayerNorm(embed_size)
            self.layer_norm_out = nn.LayerNorm(embed_size)
      
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Sequential(nn.Linear(embed_size, 16), nn.ELU(), nn.Dropout(0.2), nn.Linear(16, output_size))
    
    def forward(self, cnn_input, lin_input, batch_size, embed_size, src_mask=None):
        
        lin_x = self.linear_in(lin_input)
        cnn_x = self.cnn_embedding(cnn_input)
        cnn_x = cnn_x.view(batch_size, self.max_sequence_length, embed_size)
        
        x = torch.cat([lin_x, cnn_x], dim=-1)

        if self.layer_norm:
            x = self.layer_norm_in(x)
        x = self.dropout(x)
        x = self.positonal_embedding(x)
        if src_mask is None:
            x = self.encoder(x)
        else:
            x = self.encoder(x, mask=src_mask)
        x = self.linear_out(x)
        if self.layer_norm:
            x = self.layer_norm_out(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
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

        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(4, out_channels, kernel_size=kernel_size, stride=1)

        self.batchnorm1 = nn.BatchNorm2d(4)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=3)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(4)
        
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        # x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        # x = self.max_pool2(x)
        x = self.dropout2(x)
        x = self.global_avg_pool(x)
        x = x.flatten(start_dim=1)
        return x
    
# for tests
if __name__ == "__main__":

    transformer = TactileTransformer(33, 9, 32, 5, 256, 256, 2, 100, 2)

    lin_x = torch.randn(2, 100, 33)
    cnn_x = torch.randn(2, 100, 9, 64, 64)
    cnn_x = cnn_x.view(2*100, 9, 64, 64)

    x = transformer(cnn_x, lin_x, 2)
    print(x.shape)