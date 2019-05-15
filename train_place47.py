from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as trn

from torch.utils.data import DataLoader
from torch.autograd import variable as V

from torchvision import models
from collections import OrderedDict
import csv
import torch
import shutil

import nets

import os
import sys

from utils.utils import save_checkpoint, adjust_learning_rate, init_logger
from stage import train, validate, validate_video
from datasets import ListFromTxt,SceneImageFolder

PLACE47_TRAIN_FILE = 'data/place47_train.txt'
PLACE47_TRAIN_ROOT = '/data/place/data_large'

PLACE47_VALID_FILE = 'data/place47_valid.txt'
PLACE47_VALID_ROOT = '/data/place/val_256'

VIDEO_ROOT = '/data/korea/movie/New_world'
SCENE_TO_CLASS_CSV = VIDEO_ROOT + '.csv'
# MODEL_CKPT = 'ckpts/resnet50_latest.pth.tar'

EPOCH = 30
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

TRAIN_PRINT_FREQ = 600
VALID_PRINT_FREQ = 3
VIDEO_PRINT_FREQ = 3

SAVE = 'resnet50'


def main():
    start_epoch = 0
    best_prec1 = 0
    log = init_logger('logs/{}.txt'.format(SAVE))
    model = nets.Resnet50(47)
    for n, p in model.named_modules():
        if isinstance(p, torch.nn.Linear):
            torch.nn.init.xavier_normal(p.weight)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_trn = trn.Compose([
        trn.RandomSizedCrop(224),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        normalize
    ])
    valid_trn = trn.Compose([
        trn.CenterCrop(224),
        trn.ToTensor(),
        normalize
    ])
    train_loader = DataLoader(ListFromTxt(PLACE47_TRAIN_FILE, PLACE47_TRAIN_ROOT, train_trn)
                              , batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(ListFromTxt(PLACE47_VALID_FILE, PLACE47_VALID_ROOT, valid_trn)
                              , batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    video_loader = DataLoader(SceneImageFolder(VIDEO_ROOT, SCENE_TO_CLASS_CSV, valid_trn)
                              , batch_size=512, shuffle=False, num_workers=4, pin_memory=True)


    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(start_epoch, EPOCH):
        # adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, TRAIN_PRINT_FREQ)
        # evaluate on validation set
        prec1 = validate(valid_loader, model, criterion, VALID_PRINT_FREQ)
        # evaluate on validation video set
        validate_video(video_loader, model, criterion, VIDEO_PRINT_FREQ)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, 'ckpts/{}_ep{}'.format(SAVE,epoch))


    # validate(valid_loader, model, criterion)


if __name__ == '__main__':
    a='1003'.zfill(2)
    print(a)

    #main()
