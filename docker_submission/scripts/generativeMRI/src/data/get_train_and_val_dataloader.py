import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import Dataset, DataLoader
from pathlib import Path


import os



class CustomDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data, transform)

    def __getitem__(self, index):
        out = self._transform(index)
        out['path'] = self.data[index]['image']

        return out
        
def get_data_dicts_mood(data_dir, args):
    
    data_list = sorted(os.listdir(data_dir))
    data_dicts = [{"image": (os.path.join(data_dir, name))} for name in data_list]
    print( len(data_dicts), 'images used from', data_dir)
    return data_dicts


def get_data_loader(data_dir, args, shuffle=True, drop_last=False):

    tf = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]) if args.is_grayscale else lambda x: x,
            # transforms.Resized(keys=["image"], spatial_size=args.isize),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    data_dicts = get_data_dicts_mood(data_dir, args)
    
    ds = CustomDataset(data=data_dicts, transform=tf)


    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=shuffle,
                        num_workers=args.num_workers,
                        drop_last=drop_last,
                        pin_memory=False)

    return loader


