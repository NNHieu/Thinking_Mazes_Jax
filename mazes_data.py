""" mazes_data.py
    Maze related dataloaders
    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.
    Developed for DeepThinking project
    October 2021
"""

import os
from typing import Callable, Optional
import torch
from torch.utils import data
from easy_to_hard_data import download_url, extract_zip
from jax import numpy as jnp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST

class MazeDataset(torch.utils.data.Dataset):
    """This is a dataset class for mazes.
    padding and cropping is done correctly within this class for small and large mazes.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 size: int = 9,
                 transform: Optional[Callable] = None,
                 download: bool = True):

        self.root = root
        self.train = train
        self.size = size
        self.transform = transform

        self.folder_name = f"maze_data_{'train' if self.train else 'test'}_{size}"
        url = f"https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/" \
              f"{self.folder_name}.tar.gz"

        if download:
            self.download(url)

        print(f"Loading mazes of size {size} x {size}.")

        inputs_path = os.path.join(root, self.folder_name, "inputs.npy")
        solutions_path = os.path.join(root, self.folder_name, "solutions.npy")
        inputs_np = jnp.load(inputs_path)
        targets_np = jnp.load(solutions_path)

        # self.inputs = torch.from_numpy(inputs_np).float()
        # self.targets = torch.from_numpy(targets_np).long()
        self.inputs = inputs_np.astype(np.float32)
        self.targets = targets_np.astype(np.int32)


    def __getitem__(self, index):
        img, target = self.inputs[index], self.targets[index]

        if self.transform is not None:
            stacked = torch.cat([img, target.unsqueeze(0)], dim=0)
            stacked = self.transform(stacked)
            img = stacked[:3].float()
            target = stacked[3].long()

        return img, target

    def __len__(self):
        return self.inputs.shape[0]

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.folder_name)
        if not os.path.exists(fpath):
            return False
        return True

    def download(self, url) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        path = download_url(url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return jnp.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return jnp.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return jnp.ravel(jnp.array(pic, dtype=jnp.float32))

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_maze_loader(root, train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    train_data = MazeDataset(root, train=True, size=train_data, download=True)
    testset = MazeDataset(root, train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = NumpyLoader(trainset,
                            num_workers=0,
                            batch_size=train_batch_size,
                            shuffle=shuffle,
                            drop_last=True)
    valloader = NumpyLoader(valset,
                            num_workers=0,
                            batch_size=test_batch_size,
                            shuffle=False,
                            drop_last=False)
    testloader = NumpyLoader(testset,
                            num_workers=0,
                            batch_size=test_batch_size,
                            shuffle=False,
                            drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders