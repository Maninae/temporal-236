import random
import os
from os.path import join, isdir, isfile, basename

import torch
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


    def __init__(self, directory, cut_threshold=15000, transforms=None, debug=False):
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
        
        self.len = sum(map(len, self.selectable_dict.values())) # Sum of the len() of selectable indices for all dirs
        self.dprint("len(self):", self.len)

        self.transforms = transforms if transforms is not None else AnimatedDataset._default_transforms
        # self.research_stream.close()


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


    def _tensor_from_img_file(self, filepath):
        image = Image.open(filepath).convert("L")
        return transforms.ToTensor()(image)


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
                three_images = map(self.transforms, map(self._tensor_from_img_file, files[index:index+3]))
                
                # Apply our data augmentation transforms (horz flip) with a 50% chance. 
                should_hflip = random.random() < 0.5
                if should_hflip:
                    three_images = map(transforms.RandomHorizontalFlip(p=1.), three_images)
                    
                before, current, after = three_images
                break

            query -= len(selectable_indices)
        
        x = (before, after)
        y = current
        return (x, y)


