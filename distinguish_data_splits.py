import os
from random import random
from util.datasets import dataset_factory


if __name__ == "__main__":

    for dsname in ("ocean", "animated", "breakout"):
        ds = dataset_factory(dsname)
        print("Dataset length %d" % len(ds))

        train_threshold = 0.9
        val_threshold = 0.95

        train = []
        val = []
        test = []
        for i in range(len(ds)):
            rand = random()
            if rand < train_threshold:
                train.append(i)
            elif rand < val_threshold:
                val.append(i)
            else:
                test.append(i)
        splits_dict = {
            "train" : train,
            "val" : val,
            "test" : test
        }
        for split in ("train", "test", "val"):
            with open("data/{}_{}_indices.txt".format(dsname, split), "w") as f:
                f.write("\n".join(map(str, splits_dict[split])))


