# coding=utf-8

# Helper script to split the images in a given folder into train/dev/test splits

import argparse
import math
import os
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, help="Name of source folder")
parser.add_argument("--train", type=float, default=0.8, help="Proportion of training examples")
parser.add_argument("--val", type=float, default=0.1, help="Proportion of validation examples")
parser.add_argument("--test", type=float, default=0.1, help="Proportion of test examples")
args = parser.parse_args()

# Sanity check
assert(os.path.isdir(args.src))
#assert(math.isclose(args.train + args.val + args.test, 1.0))

# Create the required directories
os.makedirs("{}_train".format(args.src)) #, exist_ok=True
os.makedirs("{}_val".format(args.src)) #exist_ok=True
os.makedirs("{}_test".format(args.src)) #, exist_ok=True

# Compute number of examples in each dataset
filenames = sorted(os.listdir(args.src))
num_files = len(filenames)
num_train = int(num_files * args.train)
num_val = int(num_files * args.val)
num_test = num_files - num_train - num_val

print(num_train, num_val, num_test)
# assert(False)

for i, filename in enumerate(filenames):
    
    # Use first 80% for training
    if i < num_train:
        # shutil.copyfile("{}/{}".format(args.src, filename),
        #                 "{}_train/{}".format(args.src, filename))
        continue

    # Use next 10% for validation
    elif i < num_train + num_val:
        # shutil.copyfile("{}/{}".format(args.src, filename),
        #                 "{}_val/{}".format(args.src, filename))
        continue

    # Use last 10% for test
    else:
        shutil.copyfile("{}/{}".format(args.src, filename),
                        "{}_test/{}".format(args.src, filename))
    
# Sanity check
print(len(sorted(os.listdir("{}_train".format(args.src)))))
print(len(sorted(os.listdir("{}_val".format(args.src)))))
print(len(sorted(os.listdir("{}_test".format(args.src)))))
