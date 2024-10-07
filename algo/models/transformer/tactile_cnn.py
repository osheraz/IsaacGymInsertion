import torch
import torch.nn as nn
import torch.nn.functional as F


# Adopted from https://gist.github.com/kevinzakka/dd9fa5177cda13593524f4d71eb38ad5
class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].

    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=False):
        """Constructor.

        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        b, c, h, w = x.shape
        softmax = F.softmax(x.reshape(-1, h * w), dim=-1)

        xc, yc = self._coord_grid(h, w, x.device)

        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


# CNN architecture modified for 3 stacked grayscale tactile images of size (3, 32, 64)
class CNNWithSpatialSoftArgmax(nn.Module):
    def __init__(self, latent_dim):
        super(CNNWithSpatialSoftArgmax, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=2, padding=0),  # [3, 32, 64] -> [32, 13, 29]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),  # [32, 13, 29] -> [64, 10, 26]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),  # [64, 10, 26] -> [64, 8, 24]
            nn.ReLU(),
            SpatialSoftArgmax(normalize=True),
            nn.Linear(128, latent_dim)

        )

    def forward(self, x):
        return self.cnn(x)


if __name__ == "__main__":
    # Example usage
    model = CNNWithSpatialSoftArgmax(latent_dim=32)
    input_data = torch.randn(5, 3, 32, 64)  # Batch size 1, 3-channel input (3 stacked grayscale tactile images)
    output = model(input_data)

    print(output.shape)
