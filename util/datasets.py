# Standard imports
import os
import random
import torch
import pickle
from os.path import basename
from os.path import isdir
from os.path import isfile
from os.path import join
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


def dataset_factory(dataset, debug=False):
    """ Returns a Dataset instance for the given `dataset` name. """
    
    # TODO: Figure out how to do train/val/test split for Animated
    if dataset == "animated":
        with open("data/animated/selectable_dict.pkl", "rb") as f:
            selectable_dict = pickle.load(f)
        with open("data/animated/files_dict.pkl", "rb") as f:
            files_dict = pickle.load(f)
        return AnimatedDataset.from_dicts(files_dict, selectable_dict, debug=debug)
    if dataset == "animated_train": return AnimatedTrain()
    if dataset == "animated_val": return AnimatedVal()
    if dataset == "animated_test": return AnimatedTest()

    if dataset == "breakout": return Breakout()
    if dataset == "breakout_train": return BreakoutTrain()
    if dataset == "breakout_val": return BreakoutVal()
    if dataset == "breakout_test": return BreakoutTest()

    if dataset == "ocean": return Ocean()
    if dataset == "ocean_train": return OceanTrain()
    if dataset == "ocean_val": return OceanVal()
    if dataset == "ocean_test": return OceanTest()

    raise Exception("Invalid dataset: {}".format(dataset))
    

class GenericDataset(Dataset):
    """ Returns frame examples in the following format:

            x[i] = (frame_i, frame_i+2)
            y[i] = frame_i+1

    """

    _default_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # mean and std
    ])

    def __init__(self, directory, mode, transforms=None):
        """ Initializes the GenericDataset.

            Args:
                directory: str
                    The path to the directory containing the frame data.
                mode: str
                    The mode to use when retrieving the frame data. Should be
                    one of "L" (for 8-bit pixels in black and white) or
                    "RGB" (for 3x8-bit pixels in true color)
                transforms: transforms.Compose
                    Optional, used to transform the frame data if the default
                    transforms are not desired.

            Returns:
                None
        """
        super().__init__()
        self.directory = directory
        self.mode = mode
        self.files = sorted([filename for filename in os.listdir(directory) if filename.endswith(".png")])
        self.transforms = transforms if transforms else GenericDataset._default_transforms

    def __len__(self):
        """ Returns the size of the dataset. """
        return len(self.files) - 2

    def __getitem__(self, i):
        """ Supports integer indexing from 0 to len(self) exclusive. """
        before, current, after = map(self._tensor_from_img_file, self.files[i:i + 3])
        x = (before, after)
        y = current
        return (x, y)

    def _tensor_from_img_file(self, filename):
        filepath = join(self.directory, filename)
        image = Image.open(filepath).convert(self.mode)
        return self.transforms(image)


class Ocean(GenericDataset):
    def __init__(self):
        super().__init__("data/ocean", "RGB")

class Breakout(GenericDataset):
    def __init__(self):
        super().__init__("data/breakout", "L")


##########     The Splits      ####################

class OceanTrain(Subset):
    def __init__(self):
        with open("data/ocean_train_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        super().__init__(Ocean(), indices)


class OceanVal(Subset):
    def __init__(self):
        with open("data/ocean_val_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        super().__init__(Ocean(), indices)


class OceanTest(Subset):
    def __init__(self):
        with open("data/ocean_test_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        super().__init__(Ocean(), indices)


class BreakoutTrain(Subset):
    def __init__(self):
        with open("data/breakout_train_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        super().__init__(Breakout(), indices)
        #super().__init__("/Volumes/Seagate Backup Plus Drive/cs_236_stuff/ocean_test", "RGB")


class BreakoutVal(Subset):
    def __init__(self):
        with open("data/breakout_val_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        super().__init__(Breakout(), indices)


class BreakoutTest(Subset):
    def __init__(self):
        with open("data/breakout_test_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        super().__init__(Breakout(), indices)


######## Splits for Animated as well

class AnimatedTrain(Subset):
    def __init__(self):
        with open("data/animated_train_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        with open("data/animated/selectable_dict.pkl", "rb") as f:
            selectable_dict = pickle.load(f)
        with open("data/animated/files_dict.pkl", "rb") as f:
            files_dict = pickle.load(f)
        super().__init__(AnimatedDataset.from_dicts(files_dict, selectable_dict), indices)


class AnimatedVal(Subset):
    def __init__(self):
        with open("data/animated_val_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        with open("data/animated/selectable_dict.pkl", "rb") as f:
            selectable_dict = pickle.load(f)
        with open("data/animated/files_dict.pkl", "rb") as f:
            files_dict = pickle.load(f)
        super().__init__(AnimatedDataset.from_dicts(files_dict, selectable_dict), indices)


class AnimatedTest(Subset):
    def __init__(self):
        with open("data/animated_test_indices.txt") as f:
            indices = list(map(int, f.read().split()))
        with open("data/animated/selectable_dict.pkl", "rb") as f:
            selectable_dict = pickle.load(f)
        with open("data/animated/files_dict.pkl", "rb") as f:
            files_dict = pickle.load(f)
        super().__init__(AnimatedDataset.from_dicts(files_dict, selectable_dict), indices)

        #super().__init__("/Volumes/Seagate Backup Plus Drive/cs_236_stuff/breakout/all-frames_test", "L")

#################################################

class AnimatedDataset(Dataset):
    """ The dataset of consecutive triplets of frames from one of the animated sequences.
        There can be multiple videos, and we want to draw uniformly at random
        from all possible triplets across videos.
        Return: x=(frame_k, frame_k+2), y=(frame_k+1). Frames are tensors.
    """
    
    _default_transforms = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))]) # mean and stddev


    # Debug print
    def dprint(self, *args, **kwargs):
        if self.debug:
            return print("[AnimatedDataset]", *args, **kwargs)


    @classmethod
    def from_dicts(cls, files_dict, selectable_dict, transforms=None, debug=False):
        dataset = cls("/Volumes/Seagate Backup Plus Drive/cs_236_stuff/animated", cut_threshold=15000, transforms=transforms, reconstruct=False, debug=debug)
        dataset.dprint("Loading dicts from pickle.")
        dataset.files_dict = files_dict
        dataset.selectable_dict = selectable_dict
        dataset.len = sum(map(len, dataset.selectable_dict.values()))
        dataset.dprint("New len:", dataset.len)
        return dataset
        

    def __init__(self, directory, cut_threshold=15000, transforms=None, reconstruct=True, debug=False):
        super(AnimatedDataset, self).__init__()
        
        # Debugging: video cut heuristic
        # self.research_stream = open("png_pixel_deltas.txt", "w")

        self.debug = debug
        self.directory = directory
        # Heuristic for predicting whether there's a cut in the video between two frames.
        self.cut_threshold = cut_threshold;

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
        self.dprint("Discovered %d dirs:" % len(video_dirs), video_dirs)

        self.files_dict = {}
        self.selectable_dict = {}
        
        if reconstruct:
            self._construct_files_and_selectable_dicts(video_dirs)
        
        self.len = sum(map(len, self.selectable_dict.values())) # Sum of the len() of selectable indices for all dirs
        self.dprint("len(self):", self.len)

        self.transforms = transforms if transforms is not None else AnimatedDataset._default_transforms
        # self.research_stream.close()

    def _construct_files_and_selectable_dicts(self, video_dirs):
        # Construct the files_dict and selectable_dict
        for dirpath in video_dirs:
            files = sorted([join(dirpath, p) for p in os.listdir(dirpath) if p.endswith(".png")])
            if len(files) < 3:
                continue # Ignore this directory, not enough frames to form triplets

            self.files_dict[dirpath] = files

            self.dprint("Finding cuts / determining selectable indices for:", dirpath)
            self.selectable_dict[dirpath] = self._selectable_indices_without_cuts(files)
            
            self.dprint("%s: %d JPGs, %d selectable."
                        % (basename(dirpath), len(files), len(self.selectable_dict[dirpath])))


    def _selectable_indices_without_cuts(self, list_of_files):
        # e.g. If 3 is in |cut_indices|, then there is a cut between frame 3 and 4.
        cut_indices = []
        
        k = 0
        first = self._tensor_from_img_file(list_of_files[k])
        while k < (len(list_of_files) - 1):
            second = self._tensor_from_img_file(list_of_files[k+1])
            
            # Is this a cut?
            abs_L1_pixel_delta = torch.sum(torch.abs(first - second))
            is_cut = abs_L1_pixel_delta > self.cut_threshold
            # ..If so, i is invalid, remove the invalid index
            if is_cut:
                self.dprint("Determined cut, L1 diff:", abs_L1_pixel_delta)
                cut_indices.append(k)

            # Debugging: video cut heuristic
            # self.research_stream.write("%04f," % abs_L1_pixel_delta)
                
            k += 1
            first = second

            if k % 500 == 0:
                self.dprint("k = %d, nb cuts = %d" % (k, len(cut_indices)))

        # To generalize the algorithm for finding selectable indices, we will consider
        #  the end of the video as a cut as well.
        cut_indices.append(len(list_of_files) - 1)

        valid_indices_to_select = []
        pre = 0
        for cut_index in cut_indices:
            if pre < cut_index:
                valid_indices_to_select.extend(range(pre, cut_index - 1))
            pre = cut_index + 1

        return valid_indices_to_select


    def __len__(self):
        """ Provides the size of the dataset.
        """
        return self.len


    def _image_from_img_file(self, filepath):
        image = Image.open(filepath).convert("RGB")
        return image

    def _tensor_from_image(self, image):
        return transforms.ToTensor()(image)

    # Deprecated
    def _tensor_from_img_file(self, filepath):
        image = _image_from_img_file(filepath)
        return _tensor_from_image(image)


    def __getitem__(self, i):
        """ Supports integer indexing from 0 to len(self) exclusive.
        """
        # Use items() iterator to generate an implicit ordering for videos' frames,
        #  then simply index at the appropriate dirpath's frames

        query = i  # Avoid corrupting input parameter i

        for dirpath, files in self.files_dict.items():
            selectable_indices = self.selectable_dict[dirpath]

            if query < len(selectable_indices):
                index = selectable_indices[query]
                three_images = map(self._image_from_img_file, files[index:index+3])
                
                # Apply our data augmentation transforms (horz flip) with a 50% chance. 
                should_hflip = random.random() < 0.5
                if should_hflip:
                    three_images = map(transforms.RandomHorizontalFlip(p=1.), three_images)
                    
                before, current, after = map(self.transforms, map(self._tensor_from_image, three_images))
                break

            query -= len(selectable_indices)
        
        x = (before, after)
        y = current
        return (x, y)


