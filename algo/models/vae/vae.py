import pytorch_lightning as pl
from torch import nn
import torch
from torchvision.utils import make_grid

import numpy as np
from matplotlib.pyplot import imshow, figure

from .resnet_modules import resnet18_decoder, resnet18_encoder


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
