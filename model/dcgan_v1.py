import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """ The Generator is responsible for learning latent features that will
        allow it to generate frames that it can use to fool the Discriminator.
    """

    def __init__(self, channels):
        """ Initializes the Generator.

            Args:
                channels: int
                    Number of channels for each image in the dataset.

            Returns:
                None
        """
        super().__init__()

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

        # Maps the before and after frames to latent representations
        self.x_to_z = nn.Sequential(
            *generator_x_to_z_block(2 * channels, 32),
            *generator_x_to_z_block(32, 64),
            *generator_x_to_z_block(64, 128),
            nn.Tanh(),
        )

        # Maps latent representations to the interpolated frames
        self.z_to_y = nn.Sequential(
            *generator_z_to_y_block(128, 64),
            *generator_z_to_y_block(64, 32),
            *generator_z_to_y_block(32, channels),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.x_to_z(x)
        img = self.z_to_y(z)
        return img


class Discriminator(nn.Module):
    """ The Discriminator is responsible for determining whether a given frame
        is real or generated by the Generator.
    """

    def __init__(self, channels, img_size):
        """ Initializes the Discriminator.

            Args:
                channels: int
                    Number of channels for each image in the dataset.
                img_size: int
                    The size of the images in each spatial dimension.

            Returns:
                None
        """
        super().__init__()

        def discriminator_block(in_channels, out_channels, bn=True):
            block = [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_channels, 0.8))
            return block

        # Maps the frames to latent representations
        self.model = nn.Sequential(
            *discriminator_block(channels, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        # Computes likelihoods for each frame from latent representations
        ds_size = img_size // 8 # factor of 2 for each block
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def weights_init(m):
    """ Initializes the the weights and biases for the Generator and
        Discriminator.

        Args:
            m: Module
                A Module whose weights are to be initialized.

        Returns:
            None
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
