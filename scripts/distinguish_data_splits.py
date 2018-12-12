import os
from random import random
from util.datasets import dataset_factory


if __name__ == "__main__":
    animated_ds = dataset_factory("ocean")
    print("AD length %d" % len(animated_ds))

    prob = 200. / len(animated_ds)

    test_indices = [i for i in range(len(animated_ds)) if random() < prob]
    print("%d test indices" % len(test_indices))
    if len(test_indices) >= 200 and len(test_indices) < 205:
        print(test_indices)



