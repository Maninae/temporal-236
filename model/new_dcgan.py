import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):

    def __init__(self, channels):
        """ Initializes the Generator.

            Args:
                channels: list of int
                    Contains the channel sizes at each stage in the Generator.
                    channels[0] should be the number of channels for a single image.

            Returns:
                None
        """

        super(Generator, self).__init__(channels)

        def generator_x_to_z_block(in_channels, out_channels):
            return [
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.MaxPool2d(2),
            ]

        def generator_z_to_y_block(in_channels, out_channels):
            return [
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm2d(in_channels, 0.8),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            ]

        # Downsamples the before and after frames to latent representations
        x_to_z = generator_x_to_z_block(2 * channels[0], channels[1])
        for i in range(1, len(channels) - 1):
            x_to_z.extend(generator_x_to_z_block(channels[i], channels[i + 1]))
        self.x_to_z = nn.Sequential(x_to_z)

        # Upsamples latent representations to predicted frames
        z_to_y = []
        for i in range(len(channels) - 1, 0, -1):
            z_to_y.extend(generator_z_to_y_block(channels[i], channels[i - 1]))
        self.z_to_y = nn.Sequential(z_to_y)

    def forward(self, x):
        z = self.x_to_z(x)
        img = self.z_to_y(z)
        return img


class Discriminator(nn.Module):

    def __init__(self, channels, img_size):
    
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, bn=True):
            block = [
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            return block

        model = discriminator_block(channels[0], channels[1], bn=False)
        for i in range(1, len(channels) - 1):
            model.extend(discriminator_block(channels[i], channels[i + 1]))
        self.model = nn.Sequential(model)

        ds_size = img_size // (2 ** (len(channels) - 1))
        self.adv_layer = nn.Sequential(
            nn.Linear(channels[-1] * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
