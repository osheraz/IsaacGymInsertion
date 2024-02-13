import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.utils import make_grid

import numpy as np
from matplotlib.pyplot import imshow, figure

import torch
from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(
        Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes)
    )


def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(
        Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes)
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class EncoderBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, first_conv=False, maxpool1=False):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(
        self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False
    ):
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(
            64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False
        )

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x


def resnet18_encoder(first_conv, maxpool1):
    return ResNetEncoder(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder(
        DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1
    )

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False,
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x = batch.permute(1, 0, 2, 3, 4)
        x = x.reshape(-1, *x.shape[2:])
        # x = x.permute(0,3,1,2)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "reconstruction": recon_loss.mean(),
                "kl": kl.mean(),
            }
        )

        return {"loss": elbo, "pred": x_hat}

    def forward(self, batch):
        x = batch.permute(1, 0, 2, 3, 4)
        x = x.reshape(-1, *x.shape[2:])
        # x = x.permute(0,3,1,2)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        return x_hat, z

    def get_embedding(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def validation_step(self, batch, batch_idx):
        x = batch.permute(1, 0, 2, 3, 4)
        x = x.reshape(-1, *x.shape[2:])
        z = self.get_embedding(x)
        x_hat = self.decoder(z)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        return {"loss": recon_loss, "pred": x_hat}


class VAE_Callback(pl.Callback):
    def __init__(self, every_n_epochs):
        super().__init__()
        self.img_size = None
        self.num_preds = 8
        self.every_n_epochs = every_n_epochs

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.current_epoch % self.every_n_epochs == 0:
            x = batch.permute(1, 0, 2, 3, 4)
            x = x.reshape(-1, *x.shape[2:])
            # idx = np.random.randint(0, x.size(0)-1, self.num_preds, )
            max_len = self.num_preds if self.num_preds <= x.size(0) else x.size(0)
            idx = np.arange(max_len)
            x_org = x[idx].detach().cpu()
            x_pred = outputs["pred"][idx].detach().cpu()
            # UNDO DATA NORMALIZATION
            # mean, std = np.array(0.5), np.array(0.5)
            img_pred = make_grid(x_pred).permute(1, 2, 0).numpy()  # * std + mean
            img_org = make_grid(x_org).permute(1, 2, 0).numpy()  # * std + mean
            img = np.vstack((img_org, img_pred))
            # PLOT IMAGES
            trainer.logger.experiment.add_image(
                f"reconstruction",
                torch.tensor(img).permute(2, 0, 1),
                global_step=trainer.global_step,
            )

    # def on_train_epoch_end(self, trainer, pl_module):
    #     if trainer.current_epoch % self.every_n_epochs == 0:
    #         rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
    #         p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.ones_like(rand_v))
    #         z = p.rsample()

    #         # SAMPLE IMAGES
    #         with torch.no_grad():
    #             pred = pl_module.decoder(z.to(pl_module.device)).cpu()

    #         # UNDO DATA NORMALIZATION
    #         mean, std = np.array(0.5), np.array(0.5)
    #         img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

    #         # PLOT IMAGES
    #         trainer.logger.experiment.add_image(f'img_{trainer.current_epoch}',torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step)


import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import TactileDataset, output_map, get_buffer_paths, CircleMaskTransform, print_sensor_ids
import pandas as pd
from torch.utils.data import DataLoader

def get_val_images(num):
    idx = np.random.randint(0, val_set.__len__() - 1, num)
    in_seq = torch.stack([val_set[idx[i]]['in_seq'] for i in range(num)], dim=0)
    ee_pose = torch.stack([val_set[idx[i]]['ee_pose'] for i in range(num)], dim=0)
    return (in_seq, ee_pose)


if __name__ == '__main__':
    batch_sz = 8

    # Setting the seed
    seed = 42
    pl.seed_everything(seed)

    # check if gpu available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # train and validation dataset
    path_root = os.path.dirname(os.path.abspath("."))
    path_data = f"{path_root}/data_collection/train_data/"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    leds = 'combined'
    gel = 'combined'
    indenter = ['sphere3', 'sphere4', 'sphere5', 'square', 'hexagon', 'ellipse']

    # print_sensor_ids(leds, gel, indenter)

    buffer_paths_to_train, buffer_test_paths, sensors_1, sensors_2 = get_buffer_paths(leds, gel, indenter,
                                                                                      train_sensor_id=[3],

                                                                                      # train_sensor_id=[3, 10, 9, 18, 4, 17, 15, 16, 12, 14, 13, 11, 2, 1, 0, 7, 8, 6, 5],
                                                                                      test_sensor_id=[19])


    train_dataset = DIGIT_Dataset(root_path=path_root, data_path=path_data, transform=transform)
    n = train_dataset.__len__()
    n_train = int(n * 0.7)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(train_dataset, [n_train, n_val],
                                                       generator=torch.Generator().manual_seed(seed))
    train_loader = data.DataLoader(train_set, batch_size=batch_sz, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=batch_sz, shuffle=False, drop_last=False, num_workers=4)
    print("[OK] Train and validation dataloaders ready")

    # load digit encoder
    base_path_digit = f"{path_root}/ncf/digit_seq2seq/models_checkpoints/"
    digit_embeddings_path = f"{base_path_digit}/digit_embs_weights.pth"

    digit_embeddings_model = VAE()
    digit_embeddings_model.load_state_dict(torch.load(digit_embeddings_path, map_location=device))
    for param in digit_embeddings_model.parameters():
        param.requires_grad = False
    digit_embeddings_model.eval()
    print("[OK] Loaded DIGIT embeddings model")

    # define digit seq2seq model
    emb_sz = (20 * 30) + 7  # this is the dimension of the embeddings
    encoder = Encoder(emb_dim=emb_sz, hid_dim=emb_sz, num_layers=2, dropout=0.8).to(device)
    decoder = Decoder(emb_dim=emb_sz, hid_dim=emb_sz, num_layers=2, dropout=0.8).to(device)
    digit_seq2seq = TactilePoseEncoder(digit_embeddings_model, encoder, decoder).to(device)
    print("[OK] DIGIT seq2seq autoencoder defined")

    # create train data logger
    logger = TensorBoardLogger('train_log', name=f'digit_seq2seq')
    # create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(logger=logger,
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=51,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    VAE_Callback(get_val_images(5), every_n_epochs=5),
                                    LearningRateMonitor("epoch")]
                         )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # train external contact model
    print("[INFO] Starting DIGIT seq2seq autoencoder training ...")
    trainer.fit(digit_seq2seq, train_loader, val_loader)

    print("*** END ***")