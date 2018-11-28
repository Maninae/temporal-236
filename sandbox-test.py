from util.data_loader import AnimatedDataset
from util.paths import sequences_dir
from os.path import join
import pickle

if __name__ == "__main__":
    animated_dir = join(sequences_dir, "animated")
    ds = AnimatedDataset(animated_dir, debug=True)
    with open(join(animated_dir, "AnimatedDataset.pkl"), "wb") as f:
        pickle.dump(ds, f)
    
    print(len(ds))
    print(ds[247][0][0].shape)