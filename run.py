"""
This code is inspirefd by:
eriklindernoren/PyTorch-GAN/implementations/dcgan/dcgan.py
for our CS 236 project on video frame interpolation.

URL:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
"""

# Standard imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

# Custom imports
from model.dcgan import Discriminator
from model.dcgan import Generator
from model.dcgan import weights_init
from util.datasets import dataset_factory


# Load config from file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="util/config.yaml", 
                    help="path to config file")
args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f)
    pprint.pprint(config)

# Create any required directories if necessary
os.makedirs("samples", exist_ok=True)
os.makedirs("samples/{}".format(config["dataset"]), exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/{}".format(config["dataset"]), exist_ok=True)

# Specify device to use for training / evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify default tensor type
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Initialize loss function
adversarial_loss = torch.nn.BCELoss().to(device)

# Initialize generator
generator = Generator(config["channels"]).to(device)
generator.apply(weights_init)

# Initialize discriminator
discriminator = Discriminator(config["channels"], config["img_size"]).to(device)
discriminator.apply(weights_init)

# Initialize optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), 
                               lr=config["lr"], 
                               betas=(config["b1"], config["b2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                               lr=config["lr"], 
                               betas=(config["b1"], config["b2"]))

# Initialize dataloader
dataset = dataset_factory(config["dataset"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# ----------
#  Training
# ----------

# TODO: Add plots for discriminator and generator losses
# TODO: Split datasets into train and val
# TODO: Modify train loop to also evaluate on val
# TODO: Update code to conform to most recent PyTorch version

# Track losses for each batch
d_losses = []
g_losses = []

for epoch in range(config["n_epochs"]):
    for i, (x, y) in enumerate(dataloader):

        # -------
        #  Setup
        # -------

        # Inputs and outputs
        inputs = Variable(torch.cat(x, 1).type(Tensor))
        real_imgs = Variable(y.type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(y.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(y.shape[0], 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        # Loss measures generator's ability to fool the discriminator
        gen_imgs = generator(inputs)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Loss measures discriminator's ability classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2 
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Print progress
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % 
            (epoch, config["n_epochs"], i, len(dataloader), d_loss.item(), 
             g_loss.item())
        )

        # Store losses for plotting
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        # Generate sample output
        batches_done = epoch * len(dataloader) + i
        if batches_done % config["sample_interval"] == 0:
            save_image(
                gen_imgs.data[:25], 
                "samples/{}/{}.png".format(config["dataset"], batches_done)
                nrow=5, normalize=True
            )
            # TODO: Add code to plot losses (and other metrics) here

    # Save the state of the model
    torch.save((generator.state_dict,
                discriminator.state_dict,
                optimizer_G.state_dict,
                optimizer_D.state_dict), 
                "checkpoints/{}/checkpoint_{}.pth".format(config["dataset"], epoch))
        
