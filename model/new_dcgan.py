import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from util.data_loader import BreakoutDataset

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument("--img_width", type=int, default=256, help="width of each image")
parser.add_argument("--img_height", type=int, default=256, help="height of each image")
# parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Downsamples the before and after frames to latent representations
        self.x_to_z = nn.Sequential(

            nn.BatchNorm2d(2 * opt.channels),
            nn.Conv2d(2 * opt.channels, 20, 3, stride=1, padding=1),
            nn.MaxPool2d(2), 
            
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 40, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            
            nn.BatchNorm2d(40),
            nn.Conv2d(40, 80, 3, stride=1, padding=1),
            nn.MaxPool2d(2),
            
            nn.Tanh()
        )

        # Upsamples latent representations to predicted frames 
        self.z_to_y = nn.Sequential(

            nn.BatchNorm2d(80),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(80, 40, 3, stride=1, padding=1),

            nn.BatchNorm2d(40, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(40, 20, 3, stride=1, padding=1),
            
            nn.BatchNorm2d(20, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(20, opt.channels, 3, stride=1, padding=1),
            
            nn.Tanh()
        )

    def forward(self, x):
        z = self.x_to_z(x)
        img = self.z_to_y(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # Changed the number of discriminator blocks to match number of upsampling blocks
        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 20, bn=False),
            *discriminator_block(20, 40),
            *discriminator_block(40, 80),
        )

        # The height and width of downsampled image
        ds_width = opt.img_width // 8 
        ds_height = opt.img_height // 8 
        self.adv_layer = nn.Sequential(nn.Linear(80 * ds_width * ds_height, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

