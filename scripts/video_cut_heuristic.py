import torch
from matplotlib import pyplot as plt
import numpy as np

def read_pixel_deltas(filename):
    with open(filename, "r") as f:
        splitted = f.read().split(",")
    stripped = [x.strip() for x in splitted]
    numbers = [float(x) for x in stripped if x != ""]
    return numbers

def create_and_save_histogram(deltas, savepath, nb_buckets=50):
    plt.hist(deltas, nb_buckets, facecolor="blue")
    plt.xlabel("L1 pixel differences")
    plt.ylabel("Count")
    # plt.yscale("log")
    
    plt.grid(True)
    plt.savefig(savepath)
    plt.show()
    



if __name__ == "__main__":
    deltas = read_pixel_deltas("pngformat_pixel_deltas.txt")
    #  print(np.argsort(np.argsort(deltas)) * 100 / (len(deltas) - 1)) # Get percentiles
    print(np.percentile(deltas, 99))
    quit()
    
    create_and_save_histogram(deltas, "pngformat_pd_histogram.png")