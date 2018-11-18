import os
from os.path import join

from torch.utils.data import Dataset, DataLoader
from util.paths import sequences_dir

from PIL import Image
from torchvision import transforms



class BreakoutDataset(Dataset):
    """ Specifically for breakout as a simple case right now,
        but can generalize easily to other problems. In fact, we
        eventually want to utilize the train/val separation under
        utils.paths.data_dir, not read everything from the
        "sequences-raw" directory.

        We want to return x=(frame_k, frame_k+2), y=(frame_k+1).
    """
    
    _default_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]) # mean and stddev

    def __init__(self, transforms=None):
        super(BreakoutDataset, self).__init__()
        
        # Path is relative to repo root
        self.directory = join(sequences_dir, "breakout", "all-frames")
        self.files = sorted([filename for filename in os.listdir(self.directory) if filename.endswith(".png")])
        self.transforms = transforms if transforms is not None else BreakoutDataset._default_transforms

    def __len__(self):
        """ Provides the size of the dataset.
        """
        return len(self.files) - 2


    def _tensor_from_img_file(self, filename):
        filepath = join(self.directory, filename)
        image = Image.open(filepath).convert("L")
        return self.transforms(image)


    def __getitem__(self, i):
        """ Supports integer indexing from 0 to len(self) exclusive.
        """
        before, current, after = map(self._tensor_from_img_file, self.files[i:i+3])
        
        x = (before, after)
        y = current
        return (x, y)


        