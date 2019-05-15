from torchvision.datasets.folder import default_loader, ImageFolder, make_dataset
import torchvision.transforms as trn
from torch.utils.data import Dataset
import numpy as np
import os
import csv
import sys

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']


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
        self.root = root
        self.file = file
        f = open(file, 'r')
        l = f.readlines()
        self.samples = [[os.path.join(root, row.split(' ')[0]), int(row.split(' ')[1].rstrip())] for row in l]
        f.close()

    def __getitem__(self, index):
        path, cls = self.samples[index]
        sample = self.transform(self.loader(path))
        return sample, cls, path

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


class ListFromTxt_toy(Dataset):
    r"""
    get Dataset from txt file like
    root/xxx/yyy/zzz.jpg 00
    root/xxx/yyy/zzz.jpg 01
    root/xxx/yyy/zzz.jpg 02
    """

    def __init__(self, file, root, cnt=None, transform=None):
        super(ListFromTxt_toy, self).__init__()
        self.transform = trn.ToTensor() if transform == None else transform
        self.loader = default_loader
        self.root = root
        self.file = file
        f = open(file, 'r')
        l = f.readlines()
        self.samples = np.array([[os.path.join(root, row.split(' ')[0]), int(row.split(' ')[1].rstrip())] for row in l])
        if cnt is not None and cnt <= len(self.samples):
            idx = np.random.choice(len(self.samples), cnt, replace=False)
            self.samples = self.samples[idx, :]
        f.close()

    def __getitem__(self, index):
        path, cls = self.samples[index]
        sample = self.transform(self.loader(path))
        return sample, cls  # , path

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


class SceneImageFolder(ImageFolder):
    def __init__(self, root, csv_file, transform=None, loader=default_loader):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = root
        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform

        self.imgs = self.samples
        self.scene2class = [int(row[1]) for row in csv.reader(open(csv_file, 'r'))]

    def __getitem__(self, index):
        path, scene = self.samples[index]
        sample = self.loader(path)
        target = self.scene2class[scene]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path, scene

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort(key=int)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


if __name__ == '__main__':
    # dt = ListFromTxt('data/place47_train.txt', '/data/place/data_large')
    dt = ListFromTxt_toy('data/place47_train.txt', '/data/place/data_large', cnt=1000)
    a = dt.__getitem__(1)
    print(a)
    print(dt)
