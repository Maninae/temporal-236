import os
from os.path import join, isdir, basename

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
        
        # TODO: Change back to "all-framnes" or migrate completely to "square-frames{"

        self.directory = join(sequences_dir, "breakout", "square-frames")
        # self.directory = join(sequences_dir, "breakout", "all-frames")
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
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]) # mean and stddev

    _augmented_transforms = transforms.Compose([
        # Data augmentation? horizontal flip but ALL 3 frames in triplet only
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]) # mean and stddev

    # Debug print
    def dprint(self, *args, **kwargs):
        if self.debug:
            return print("[AnimatedDataset]", *args, **kwargs)


    
    def __init__(self, directory, transforms=None, debug=False):
        super(AnimatedDataset, self).__init__()
        
        self.debug = debug
        self.directory = directory
        self.dprint("Creating dataset from dir:", self.directory)

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
        video_dirs = [join(self.directory, p) for p in os.listdir(self.directory) if isdir(join(self.directory, p))]
        self.dprint("Discovered video dirs:", video_dirs)

        self.files_dict = {}
        for dirpath in video_dirs:
            files = sorted([join(dirpath, p) for p in os.listdir(dirpath) if p.endswith(".jpg")])
            if len(files) < 3:
                continue # Ignore this directory, not enough frames to form triplets

            self.files_dict[dirpath] = files
            self.dprint("%s: %d JPGs." % (basename(dirpath), len(files)))
        
        self.len = sum([len(files) - 2 for files in self.files_dict.values()])
        self.dprint("len(self):", self.len)

        self.transforms = transforms if transforms is not None else AnimatedDataset._default_transforms

    def __len__(self):
        """ Provides the size of the dataset.
        """
        return self.len


    def _tensor_from_img_file(self, filepath):
        image = Image.open(filepath).convert("L")
        return self.transforms(image)


    def __getitem__(self, i):
        """ Supports integer indexing from 0 to len(self) exclusive.
        """
        # Use items() iterator to generate an implicit ordering for videos' frames, then simply index
        for dirpath, files in self.files_dict.items():
            if i < len(files) - 2:
                before, current, after = map(self._tensor_from_img_file, files[i:i+3])
                break
            i -= len(files) - 2
        
        x = (before, after)
        y = current
        return (x, y)

