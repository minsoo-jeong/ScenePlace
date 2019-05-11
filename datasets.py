from torchvision.datasets.folder import default_loader
from torchvision.datasets import DatasetFolder
import torchvision.transforms as trn
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset

from random import shuffle
import numpy as np
from PIL import Image
import os


class ListFromTxt(Dataset):
    r"""
    get Dataset from txt file like
    root/xxx/yyy/zzz.jpg 00
    root/xxx/yyy/zzz.jpg 01
    root/xxx/yyy/zzz.jpg 02
    """

    def __init__(self, file, root, transform=None):
        super(ListFromTxt, self).__init__()
        self.transform = trn.ToTensor() if transform == None else transform
        self.loader = default_loader
        self.root=root
        self.file=file
        f = open(file, 'r')
        l = f.readlines()
        self.samples = [[os.path.join(root, row.split(' ')[0]), int(row.split(' ')[1].rstrip())] for row in l]
        f.close()

    def __getitem__(self, index):
        path, cls = self.samples[index]
        sample = self.transform(self.loader(path))
        return sample, cls #, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Data Root: {}\n'.format(self.root)
        fmt_str += '    Data File: {}\n'.format(self.file)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__=='__main__':
    dt = ListFromTxt('data/place47_train.txt', '/data/place/data_large')
    a=dt.__getitem__(1)
    print(a)
    print(dt)
