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
from jax import numpy as jnp
import jax
import numpy as np

import errno
import os
import os.path
import tarfile
import urllib.request as ur
from typing import Optional, Callable

import numpy as np
from tqdm import tqdm

GBFACTOR = float(1 << 30)

def extract_zip(path, folder):
    file = tarfile.open(path)
    file.extractall(folder)
    file.close

def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e

def download_url(url, folder):
    filename = url.rpartition('/')[2]
    path = os.path.join(folder, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('Using existing file', filename)
        return path
    print('Downloading', url)
    makedirs(folder)
    # track downloads
    ur.urlopen(f"http://avi.koplon.com/hit_counter.py?next={url}")
    data = ur.urlopen(url)
    size = int(data.info()["Content-Length"])
    chunk_size = 1024*1024
    num_iter = int(size/chunk_size) + 2

    downloaded_size = 0

    try:
        with open(path, 'wb') as f:
            pbar = tqdm(range(num_iter))
            for i in pbar:
                chunk = data.read(chunk_size)
                downloaded_size += len(chunk)
                pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
                f.write(chunk)
    except:
        if os.path.exists(path):
             os.remove(path)
        raise RuntimeError('Stopped downloading due to interruption.')

    return path



class MazeDataset(object):
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

        # if self.transform is not None:
        #     stacked = torch.cat([img, target.unsqueeze(0)], dim=0)
        #     stacked = self.transform(stacked)
        #     img = stacked[:3].float()
        #     target = stacked[3].long()

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

def prepare_maze_loader(root, train_data, test_data):
    train_data = MazeDataset(root, train=True, size=train_data, download=True)
    test_data = MazeDataset(root, train=False, size=test_data, download=True)
    indices = np.random.permutation(train_data.inputs.shape[0])
    train_split = int(0.8 * len(train_data))
    training_idx, val_idx = indices[:train_split], indices[train_split:]
    trainset = (train_data.inputs[training_idx, :], train_data.targets[training_idx, :])
    valset = (train_data.inputs[val_idx, :], train_data.targets[val_idx, :])
    testset = (test_data.inputs, test_data.targets)
    return {"train": trainset, "val": valset, "testset": testset}

class DataLoader(object):
  def __init__(self, ds, batch_size) -> None:
    self.__ds = ds
    self.batch_size = batch_size
    self.__perms = None
  
  @property
  def batch_size(self):
    return self.__batch_size
  
  @batch_size.setter
  def batch_size(self, value):
    self.__batch_size = value
    self.__train_ds_size = len(self.__ds[0])
    self.__steps_per_epoch = self.__train_ds_size // self.__batch_size

  def new_perms(self, rng):
    self.__perms = jax.random.permutation(rng, self.__train_ds_size)
    self.__perms = self.__perms[:self.__steps_per_epoch * self.batch_size]  # skip incomplete batch
    self.__perms = self.__perms.reshape((self.__steps_per_epoch, self.batch_size))
  
  def __iter__(self):
    assert self.__perms is not None
    self.__iter_idx = 0
    return self
  
  def __next__(self):
    if self.__iter_idx >= len(self.__perms):
      raise StopIteration
    else:
      out =  [v[self.__perms[self.__iter_idx], ...] for v in self.__ds]
      self.__iter_idx += 1
      return out

  def __len__(self):
    return self.__steps_per_epoch