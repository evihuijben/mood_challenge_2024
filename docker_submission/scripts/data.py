import pandas as pd
import torch.distributed as dist
from monai import transforms
# from monai.data import CacheDataset, Dataset, ThreadDataLoader, DataLoader, partition_dataset
from pathlib import Path

import torch
import torchio as tio
import os
from monai.transforms import apply_transform
import collections.abc
from collections.abc import Callable, Sequence
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list of str): List of file paths to NIfTI images.
            transform (callable, optional): A function/transform to apply to the image.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Load NIfTI image
        image = nib.load(image_path).get_fdata()
        
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalization

        # Convert to PyTorch tensor
        image = torch.tensor(image).float()
        
        # Add channel dimension if needed
        if len(image.shape) == 3:  # Assuming image is 3D
            image = image.unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return {"image": image, "path": image_path}



def get_data_loader(data_dir, args, shuffle=True, drop_last=False):
    if args.region == 'brain':
        # transform = tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1))])  
        transform = None
    else:
        transform =  tio.Compose([tio.Resize((256, 256, 256))])

    
    data_list = sorted(os.listdir(data_dir))
    data_list = [os.path.join(data_dir, name) for name in data_list]
    ds = CustomDataset(image_paths=data_list, transform=transform)


    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                        num_workers=args.num_workers,
                        drop_last=drop_last,
                        pin_memory=False)

    return loader




def upsample(img, interpolation=''):
    img=torch.from_numpy(img)[None]
    if interpolation == '':
        transform = tio.Resize((512, 512, 512))
    else:
        transform = tio.Resize((512, 512, 512), image_interpolation=interpolation)
    new_img = transform(img)
    new_img = new_img.squeeze().numpy()
    return new_img
    