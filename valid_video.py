from torchvision.transforms import transforms as trn
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader

from datasets import ListFromTxt, SceneImageFolder
from utils.utils import init_logger
from stage import validate, validate_video_csv
import nets

import torch
import csv
import os

GENRE = 'drama'
VIDEO = 'Miss_Hammurabi_E09'


VIDEO_VALID_FILE = 'data/{}/{}.txt'.format(GENRE, VIDEO)
VIDEO_ROOT = '/data/korea/{}/{}'.format(GENRE, VIDEO)
SCENE_TO_CLASS_CSV = VIDEO_ROOT + '.csv'
MODEL_CKPT = 'ckpts/resnet50_ep8-latest.pth.tar'


PRINT_FREQ = 1
SAVE = 'validation-video-resnet50-rmac'

PLACE47_VALID_FILE = 'data/place47_valid.txt'
PLACE47_VALID_ROOT = '/data/place/val_256'

CATEGORY = 'data/place47_category.txt'
f = open(CATEGORY, 'rt', encoding='utf-8')
CATEGORY = f.read().split('\n')
f.close()

OUT_CSV = 'out/{}.csv'.format(VIDEO)


def main():
    init_logger('logs/{}.txt'.format(SAVE))


    model = nets.Resnet50(47).cuda()

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL_CKPT)['state_dict'])

    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    video_trn = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize
    ])
    valid_loader = DataLoader(ListFromTxt(PLACE47_VALID_FILE, PLACE47_VALID_ROOT, video_trn)
                              , batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    # video_loader = DataLoader(SceneImageFolder(VIDEO_ROOT, SCENE_TO_CLASS_CSV, video_trn)
    #                          , batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    video_loader = DataLoader(ListFromTxt(VIDEO_VALID_FILE, '', video_trn)
                              , batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # prec1 = validate(valid_loader, model, criterion, 30)
    prec1 = validate_video_csv(video_loader, model, criterion, CATEGORY, OUT_CSV, PRINT_FREQ)


if __name__ == '__main__':
    main()
