import numpy as np
import random
import torch
import os
from torch import nn
from torchvision import transforms


class ImageTransform:
    def __init__(self, image_transform=None):
        self.image_transform = image_transform

    def __call__(self, img_input):
        # img_input shape: [B, T, C, H, W]
        B, T, C, H, W = img_input.shape
        img_input = img_input.view(-1, C, H, W)  # Shape: [B * T, C, H, W]
        if self.image_transform is not None:
            img_input = self.image_transform(img_input)

        img_input = img_input.view(B, T, C, *img_input.shape[2:])  # Reshape back to [B, T, C, new_H, new_W]
        return img_input


class TactileTransform:
    def __init__(self, tactile_transform=None):
        self.tactile_transform = tactile_transform
        self.out_channel = 3
        self.to_gray = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()  # Convert PIL Image back to tensor
        ])

    def __call__(self, tac_input):
        # tac_input shape: [B, T, Num_cam, C, H, W]
        B, T, Num_cam, C, H, W = tac_input.shape
        tac_input = tac_input.view(-1, C, H, W)  # Shape: [B * T * Num_cam, C, H, W]

        transformed_list = []

        for i in range(tac_input.shape[0]):
            if self.out_channel == 1:
                tactile_input_reshaped = self.to_gray(tac_input[i])
                transformed_image = self.tactile_transform(tactile_input_reshaped)
            else:
                transformed_image = self.tactile_transform(tac_input[i])

            transformed_list.append(transformed_image)
        tac_input = torch.stack(transformed_list)

        tac_input = tac_input.view(B, T, Num_cam, self.out_channel,
                                   *tac_input.shape[2:])  # Reshape back to [B, T, Num_cam, C, new_H, new_W]

        return tac_input


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def define_transforms(channel, color_jitter, width, height, crop_width,
                      crop_height, img_patch_size, img_gaussian_noise=0.0, img_masking_prob=0.0):
    # Use color jitter to augment the image
    if color_jitter:
        if channel == 3:
            # no depth
            downsample = nn.Sequential(
                transforms.Resize(
                    (width, height),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ColorJitter(brightness=0.1),
            )
        else:
            # with depth, only jitter the rgb part
            downsample = lambda x: transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )(
                torch.concat(
                    [transforms.ColorJitter(brightness=0.1)(x[:, :3]), x[:, 3:]],
                    axis=1,
                )
            )

    # Not using color jitter, only downsample the image
    else:
        downsample = nn.Sequential(
            transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        )

    # Crop randomization, normalization
    transform = nn.Sequential(
        transforms.RandomCrop((crop_width, crop_height)),
    )

    # Add gaussian noise to the image
    if img_gaussian_noise > 0.0:
        transform = nn.Sequential(
            transform,
            GaussianNoise(img_gaussian_noise),
        )

    def mask_img(x):
        # Divide the image into patches and randomly mask some of them
        img_patch = x.unfold(2, img_patch_size, img_patch_size).unfold(
            3, img_patch_size, img_patch_size
        )
        mask = (
                torch.rand(
                    (
                        x.shape[0],
                        x.shape[-2] // img_patch_size,
                        x.shape[-1] // img_patch_size,
                    )
                )
                < img_masking_prob
        )
        mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
        x = x.clone()
        x.unfold(2, img_patch_size, img_patch_size).unfold(
            3, img_patch_size, img_patch_size
        )[mask] = 0
        return x

    if img_masking_prob > 0.0:
        transform = lambda x: mask_img(
            nn.Sequential(
                transforms.RandomCrop((crop_width, crop_height)),
            )(x)
        )
    # For evaluation, only center crop and normalize
    eval_transform = nn.Sequential(
        transforms.CenterCrop((crop_width, crop_height)),
    )

    print('transform {}'.format(transform))
    print('eval_transform {}'.format(eval_transform))
    print('downsample {}'.format(downsample))

    return transform, downsample, eval_transform


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


def mask_img(x, img_patch_size, img_masking_prob):
    # Divide the image into patches and randomly mask some of them
    img_patch = x.unfold(2, img_patch_size, img_patch_size).unfold(
        3, img_patch_size, img_patch_size
    )
    mask = (
            torch.rand(
                (
                    x.shape[0],
                    x.shape[-2] // img_patch_size,
                    x.shape[-1] // img_patch_size,
                )
            )
            < img_masking_prob
    )
    mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
    x = x.clone()
    x.unfold(2, img_patch_size, img_patch_size).unfold(
        3, img_patch_size, img_patch_size
    )[mask] = 0
    return x
