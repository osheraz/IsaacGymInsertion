import numpy as np
import random
import torch
import os
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt


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
    def __init__(self, tactile_transform=None, out_channel=3):
        self.tactile_transform = tactile_transform
        self.out_channel = out_channel
        self.to_gray = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()  # Convert PIL Image back to tensor
        ])

    def __call__(self, tac_input):
        # tac_input shape: [B, T, Num_cam, C, H, W]
        B, T, F, C, H, W = tac_input.shape
        tac_input = tac_input.view(-1, C, H, W)  # Shape: [B * T * F, C, H, W]

        transformed_list = []

        for i in range(tac_input.shape[0]):
            if self.out_channel == 1:
                tactile_input_reshaped = self.to_gray(tac_input[i])
                transformed_image = self.tactile_transform(tactile_input_reshaped)
            else:
                transformed_image = self.tactile_transform(tac_input[i])

            transformed_list.append(transformed_image)
        tac_input = torch.stack(transformed_list)

        tac_input = tac_input.view(B, T, F, self.out_channel,
                                   *tac_input.shape[2:])  # Reshape back to [B, T, F, C, new_H, new_W]

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
                      crop_height, img_patch_size, img_gaussian_noise=0.0, img_masking_prob=0.0, is_tactile=True):
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
    if not is_tactile:
        transform = nn.Sequential(
            CenterCropTransform((height, width), (150, 180)),
            transforms.RandomCrop((crop_width, crop_height)),
        )
    else:
        transform = nn.Sequential(
            transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
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
    if not is_tactile:
        eval_transform = nn.Sequential(
            CenterCropTransform((height, width), (150, 180)),
            # transforms.RandomCrop((crop_width, crop_height)),
        )
    else:
        eval_transform = nn.Sequential(
            transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
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

class CenterCropTransform(nn.Module):
    def __init__(self, crop_size, center):
        super().__init__()
        self.crop_size = crop_size
        self.center = center

    def forward(self, img):
        _, _, height, width = img.size()
        crop_width, crop_height = self.crop_size

        center_x, center_y = self.center

        # Ensure the center is within the bounds
        center_x = max(crop_width // 2, min(center_x, width - crop_width // 2))
        center_y = max(crop_height // 2, min(center_y, height - crop_height // 2))

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2
        right = center_x + crop_width // 2
        bottom = center_y + crop_height // 2

        img = img[:, :, top:bottom, left:right]
        return img

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


def log_output(tac_input, img_input, lin_input, out, latent, pos_rpy, save_folder, d_pos_rpy=None, session='train'):
    # Selecting the first example from the batch for demonstration
    # tac_input [B T F W H C]

    image_sequence = tac_input[0].cpu().detach().numpy()
    img_input = img_input[0].cpu().detach().numpy()
    linear_features = lin_input[0].cpu().detach().numpy()
    if d_pos_rpy is not None:
        d_pos_rpy = d_pos_rpy[0, -1, :].cpu().detach().numpy()
    pos_rpy = pos_rpy[0, -1, :].cpu().detach().numpy()

    predicted_output = out[0].cpu().detach().numpy()
    true_label = latent[0, -1, :].cpu().detach().numpy()
    # Plotting
    fig = plt.figure(figsize=(20, 10))

    # Adding subplot for image sequence (adjust as needed)
    ax1 = fig.add_subplot(2, 2, 1)
    concat_images = []
    # image_sequence [T F W H C]
    for finger_idx in range(image_sequence.shape[1]):
        finger_sequence = [np.transpose(img, (1, 2, 0)) for img in image_sequence[:, finger_idx, ...]]
        finger_sequence = np.hstack(finger_sequence)
        concat_images.append(finger_sequence)

    ax1.imshow(np.vstack(concat_images) + 0.5)  # Adjust based on image normalization
    ax1.set_title('Input Tactile Sequence')

    # Adding subplot for linear features (adjust as needed)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(d_pos_rpy[:, :], 'ok', label='hand_joints')  # Assuming the rest are actions
    # ax2.set_title('Linear input')
    # ax2.legend()

    if d_pos_rpy is not None:
        ax2 = fig.add_subplot(2, 2, 2)
        width = 0.35
        indices = np.arange(len(d_pos_rpy))
        ax2.bar(indices - width / 2, d_pos_rpy, width, label='d_pos_rpy')
        ax2.bar(indices + width / 2, pos_rpy, width, label='True Label')
        ax2.set_title('Model Output vs. True Label')
        ax2.legend()

    # Check if img_input has more than one timestep
    if img_input.ndim == 4 and img_input.shape[0] > 1:
        concat_img_input = []
        for t in range(img_input.shape[0]):
            img = img_input[t]
            img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
            img = img + 0.5  # Adjust normalization if needed
            concat_img_input.append(img)

        # Horizontally stack the images for each timestep
        concat_img_input = np.hstack(concat_img_input)
    else:
        # Handle the case where there is only one timestep
        img = img_input[0] if img_input.ndim == 4 else img_input
        img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
        img = img + 0.5  # Adjust normalization if needed
        concat_img_input = img

    # Plot the concatenated image sequence
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(concat_img_input)
    ax3.set_title('Input Image Sequence')

    # Adding subplot for Output vs. True Label comparison
    ax4 = fig.add_subplot(2, 2, 4)
    width = 0.35
    indices = np.arange(len(predicted_output))
    ax4.bar(indices - width / 2, predicted_output, width, label='Predicted')
    ax4.bar(indices + width / 2, true_label, width, label='True Label')
    ax4.set_title('Model Output vs. True Label')
    ax4.legend()

    # Adjust layout
    plt.tight_layout()
    # Saving the figure
    plt.savefig(f'{save_folder}/{session}_example.png')
    # Clean up plt to free memory
    plt.close(fig)
