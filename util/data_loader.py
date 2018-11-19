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


class AnimatedDataset(Dataset):
    """ The dataset of consecutive triplets of frames from one of the animated sequences.
        There can be multiple videos, and we want to draw uniformly at random
        from all possible triplets across videos.
        Return: x=(frame_k, frame_k+2), y=(frame_k+1). Frames are tensors.
    """
    
    _default_transforms = transforms.Compose([
        # Data augmentation? horizontal flip
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]) # mean and stddev

    def __init__(self, directory, transforms=None):
        super(AnimatedDataset, self).__init__()
        
        self.directory = directory

        # The structure under self.directory should be:
        # self.directory (e.g. "{root} / data / {train|val|test}")
        #  |--> video1
        #  |      |--> 00001.jpg
        #  |      |--> 00002.jpg
        #  |      | ....
        #  |      |--> N.jpg
        #  |--> video2
        #  |      |--> 00001.jpg
        #  |      |--> ...
        # ....
        self.files_dict = {
            dirpath: sorted([p for p in os.listdir(dirpath) if p.endswith(".jpg")])
                for dirpath in os.listdir(self.directory)
                if os.path.isdir(dirpath)
        }
        
        self.len = sum([len(files) - 2 for files in self.files_dict.values()])
        self.transforms = transforms if transforms is not None else AnimatedDataset._default_transforms

    def __len__(self):
        """ Provides the size of the dataset.
        """
        return self.len


    def _tensor_from_img_file(self, filename):
        filepath = join(self.directory, filename)
        image = Image.open(filepath).convert("L")
        return self.transforms(image)


    def __getitem__(self, i):
        """ Supports integer indexing from 0 to len(self) exclusive.
        """
        # Use items() iterator to generate an implicit ordering for videos' frames, then simply index
        for dirpath, files in self.files_dict.items():
            if i < len(files) - 2:
                before, current, after = map(self._tensor_from_img_file, self.files[i:i+3])
                break
            i -= len(files) - 2
        
        x = (before, after)
        y = current
        return (x, y)


        