from torchvision.datasets.folder import has_file_allowed_extension
from datasets import SceneImageFolder, ListFromTxt
import sys
import csv
import os
import re

GERNE='movie'
VIDEO='The_Last_Blossom'
VIDEO_ROOT = '/data/korea/{}/{}'.format(GERNE,VIDEO)
SCENE_TO_CLASS_CSV = VIDEO_ROOT + '.csv'
OUT_TXT = 'data/{}/{}.txt'.format(GERNE,VIDEO)

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys(), key=natural_keys):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames, key=natural_keys):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images




def _find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort(key=natural_keys)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

classes, class_to_idx = _find_classes(VIDEO_ROOT)
samples = make_dataset(VIDEO_ROOT, class_to_idx, IMG_EXTENSIONS)

scene2class = [int(row[1]) for row in csv.reader(open(SCENE_TO_CLASS_CSV, 'r'))]
print(classes)
print(class_to_idx)
print(scene2class)
print(len(samples))

l = ['{} {}\n'.format(path, scene2class[scene]) for path, scene in samples]
print(l[0])
print(os.path.abspath(os.path.dirname(OUT_TXT)))
if not os.path.exists(os.path.abspath(os.path.dirname(OUT_TXT))):
    print(os.path.abspath(os.path.dirname(OUT_TXT)))
    os.makedirs(os.path.abspath(os.path.dirname(OUT_TXT)))

f = open(OUT_TXT, 'w')
f.writelines(l)
