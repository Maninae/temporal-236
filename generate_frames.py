# Standard imports
import argparse
import os
import pprint
import torch
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Custom imports
from model.dcgan_v1 import Generator
from util.datasets import dataset_factory


# Number of images to generate
MAX_COUNT = 10000

# Parse command line arguments
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

# Create necessary directories
os.makedirs("generated", exist_ok=True)
os.makedirs("generated/{}".format(config["dataset"]), exist_ok=True)

# Specify device to use for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify default tensor type
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Initialize generator
generator = Generator(config["channels"]).to(device)
print("Loading checkpoint file: {}".format(args.checkpoint))
state = torch.load(args.checkpoint)
generator.load_state_dict(state[0])

# Initialize dataloader
dataset = dataset_factory(config["dataset"])
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Generate images
with tqdm(total=MAX_COUNT) as pbar:
    count = 0
    for i, (x, y) in enumerate(dataloader):
        inputs = Variable(torch.cat(x, 1).type(Tensor))
        gen_imgs = generator(inputs)
        for i in range(x[0].shape[0]):
            save_image(gen_imgs[i].data,
                       "generated/{}/{}.png".format(config["dataset"], count),
                        normalize=True)
            count += 1
            pbar.update(1)
        if count > MAX_COUNT:
            break
