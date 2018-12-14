# Standard imports
import argparse
import os
from os.path import join
import pprint
import torch
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


# Custom imports
from model.dcgan_v2 import Generator  # Owen: dcgan_v1 may have made Animated checkpoints incompatible
# from model.dcgan import Generator
from util.datasets import dataset_factory


# Number of images to generate
MAX_COUNT = 500
# Owen: Testing out bugfix re: setting model to eval mode
# MAX_COUNT = 10

# 
""" Parse command line arguments

Sample Commands:
    python generate_frames.py
    python3 generate_frames.py --config util/configs/animated.yaml --checkpoint checkpoints/animated/checkpoint_199.pth
"""
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="util/configs/breakout.yaml",
                    help="path to config file")
parser.add_argument("--checkpoint", type=str, 
                    default="checkpoints/breakout/checkpoint_0.pth",
                    help="path to checkpoint file")
args = parser.parse_args()



# Load config from file
with open(args.config, "r") as f:
    config = yaml.load(f)
    pprint.pprint(config)


# Specify device to use for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Specify default tensor type
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Initialize generator
generator = Generator(config["channels"]).to(device)
print("Loading checkpoint file: {}".format(args.checkpoint))
state = torch.load(args.checkpoint, map_location="cpu")
# Get generator's state_dict() function from (G, D, Optimizer) tuple, and call it.
# Returns a state dict, and feed this into loading method
generator.load_state_dict(state[0])

# Initialize dataloader
config["dataset"] = "animated_test"
config["batch_size"] = 1



def makeTensor(filepath):
    image = Image.open(filepath).convert("RGB")
    _default_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # mean and std
    ])
    return _default_transforms(image)

dirs = ["vizdata/breakout", "vizdata/momoshiki", "vizdata/reiner", "vizdata/ocean"]
for d in dirs:
    l = list(map(makeTensor, sorted([join(d, f) for f in os.listdir(d) if f.endswith(".png")])))

    x = (l[0].unsqueeze(0), l[2].unsqueeze(0))
    y = l[1].unsqueeze(0)
    inputs = Variable(torch.cat(x, 1).type(Tensor))
    gen_imgs = generator(inputs)
    save_image(gen_imgs[0].data,
        d + "/middle.png")

quit()

# Generate images
with tqdm(total=MAX_COUNT) as pbar:
    count = 0
    for i, (x, y) in enumerate(dataloader):
        
        for i in range(x[0].shape[0]):
            save_image(gen_imgs[i].data,
                       "generated/{}/{:05d}.png".format(config["dataset"], count),
                        normalize=True)
            save_image(y[i].data,
                       "real/{}/{:05d}.png".format(config["dataset"], count),
                       normalize=True)
            count += 1
            pbar.update(1)
        if count > MAX_COUNT:
            break
