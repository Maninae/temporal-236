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
import pickle
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Custom imports
from model.dcgan_v1 import Discriminator
from model.dcgan_v1 import Generator
from model.dcgan_v1 import weights_init
from util.datasets import dataset_factory


# Load config from file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="util/configs/breakout.yaml", 
                    help="path to config file")
args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f)
    pprint.pprint(config)

# Create any required directories if necessary
os.makedirs("samples", exist_ok=True)
os.makedirs("samples/{}_train".format(config["dataset"]), exist_ok=True)
os.makedirs("samples/{}_val".format(config["dataset"]), exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints/{}".format(config["dataset"]), exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("plots/{}_train".format(config["dataset"]), exist_ok=True)
os.makedirs("plots/{}_val".format(config["dataset"]), exist_ok=True)

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
trainname = "{}_train".format(config["dataset"])
trainset = dataset_factory(trainname, debug=True)
trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)

valname = "{}_val".format(config["dataset"])
valset = dataset_factory(valname, debug=True)
valloader = DataLoader(valset, batch_size=config["batch_size"], shuffle=True)

"""
# Load dataset from pickle, or call factory to construct one
dataset = dataset_factory(dataset_name, debug=True)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
"""

# ----------
#  Training
# ----------

# TODO: Update code to conform to most recent PyTorch version

# Track losses for each batch
d_losses_train = []
g_losses_train = []
d_losses_val = []
g_losses_val = []

for epoch in range(config["n_epochs"]):

    # ----------
    #  TRAINING
    # ----------

    # Put models into train mode
    generator.train()
    discriminator.train()

    # Process all examples in training set
    for i, (x, y) in enumerate(trainloader):

        # -------
        #  Setup
        # ------- 

        # Inputs and outputs
        inputs = Variable(torch.cat(x, 1).type(Tensor))
        real_imgs = Variable(y.type(Tensor))

        # Adversarial ground truths
        real = Variable(Tensor(y.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(y.shape[0], 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train generator
        # -----------------

        # Loss measures generator's ability to fool the discriminator
        fake_imgs = generator(inputs)
        g_loss_train = adversarial_loss(discriminator(fake_imgs), real)
        optimizer_G.zero_grad()
        g_loss_train.backward()
        optimizer_G.step()
        g_losses_train.append(g_loss_train.item())

        # ---------------------
        #  Train Discriminator 
        # ---------------------

        # Loss measures discriminator's ability to discern real from generator samples
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
        d_loss_train = (real_loss + fake_loss) / 2
        optimizer_D.zero_grad()
        d_loss_train.backward()
        optimizer_D.step()
        d_losses_train.append(d_loss_train.item())

        # Print progress
        print("[Training] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, config["n_epochs"], i, len(trainloader), d_loss_train.item(),
              g_loss_train.item()))

        # Generate sample output
        batches_done = epoch * len(trainloader) + i
        if batches_done % config["train_interval"] == 0:
            save_image(fake_imgs.data[:25], 
                       "samples/{}_train/{}.png".format(config["dataset"], batches_done),
                       nrow=5, normalize=True)

            # Plot the discriminator loss for the training set
            plt.plot(np.arange(batches_done + 1), d_losses_train)
            plt.xlabel("batches done")
            plt.ylabel("loss")
            plt.title("training loss for discriminator")
            plt.savefig("plots/{}_train/d_{}.png".format(config["dataset"], batches_done))
            plt.clf()

            # Plot the generator loss for the training set
            plt.plot(np.arange(batches_done + 1), g_losses_train)
            plt.xlabel("batches done")
            plt.ylabel("loss")
            plt.title("training loss for generator")
            plt.savefig("plots/{}_train/g_{}.png".format(config["dataset"], batches_done))

        # Delete all references (so they can be released)
        del inputs
        del real_imgs
        del fake_imgs
        del g_loss_train
        del real_loss
        del fake_loss
        del d_loss_train
        torch.cuda.empty_cache() 

    # Save the state of the model
    torch.save((generator.state_dict(),
               discriminator.state_dict(),
               optimizer_G.state_dict(),
               optimizer_D.state_dict()),
               "checkpoints/{}/checkpoint_{}.pth".format(config["dataset"], epoch))

    if config["dataset"] == "animated":
        continue
    # ------------
    #  VALIDATION
    # ------------
    # Put models into eval mode
    generator.eval()
    discriminator.eval()

    # Process all examples in validation set
    for i, (x, y) in enumerate(valloader):

        # -------
        #  Setup 
        # -------

        # Inputs and outputs
        inputs = Variable(torch.cat(x, 1).type(Tensor))
        real_imgs = Variable(y.type(Tensor))

        # Adversarial ground truths
        real = Variable(Tensor(y.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(y.shape[0], 1).fill_(0.0), requires_grad=False)

        # --------------------
        #  Evaluate Generator
        # --------------------

        fake_imgs = generator(inputs)
        g_loss_val = adversarial_loss(discriminator(fake_imgs), real)
        g_losses_val.append(g_loss_val.item())

        # ------------------------
        #  Evaluate Discriminator
        # ------------------------

        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
        d_loss_val = (real_loss + fake_loss) / 2
        d_losses_val.append(d_loss_val.item())

        # Print progress
        print("[Validation] [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, config["n_epochs"], i, len(valloader), d_loss_val.item(),
              g_loss_val.item()))

        # Generate sample output
        batches_done = epoch * len(valloader) + i
        if batches_done % config["val_interval"] == 0:
            save_image(fake_imgs.data[:25], 
                       "samples/{}_val/{}.png".format(config["dataset"], batches_done),
                       nrow=5, normalize=True)
            
            # Plot the discriminator losses on the validation set
            plt.plot(np.arange(batches_done + 1), d_losses_val)
            plt.xlabel("batches done")
            plt.ylabel("loss")
            plt.title("validation loss for discriminator")
            plt.savefig("plots/{}_val/d_{}.png".format(config["dataset"], batches_done))
            plt.clf()

            # Plot the generator losses on the validation set
            plt.plot(np.arange(batches_done + 1), g_losses_val)
            plt.xlabel("batches done")
            plt.ylabel("loss")
            plt.title("validation loss for generator")
            plt.savefig("plots/{}_val/g_{}.png".format(config["dataset"], batches_done))
            plt.clf()

        # Delete variable references so GPU memory cache can be released
        del inputs
        del real_imgs
        del fake_imgs
        del g_loss_val
        del real_loss
        del fake_loss
        del d_loss_val
        torch.cuda.empty_cache()

    # Save the state of the model
    torch.save((generator.state_dict(),
               discriminator.state_dict(),
               optimizer_G.state_dict(),
               optimizer_D.state_dict()),
               "checkpoints/{}/checkpoint_{}.pth".format(config["dataset"], epoch))
