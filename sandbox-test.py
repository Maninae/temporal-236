from util.datasets import dataset_factory
from os.path import join
import pickle

if __name__ == "__main__":
    for ds in ("animated", "ocean", "breakout"):
        for split in ("", "_train", "_test", "_val"):
            dsname = ds + split
            print("dsname:", dsname)
            print(len(dataset_factory(dsname)))

