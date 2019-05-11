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
from stage import train, validate
from datasets import ListFromTxt

PLACE47_TRAIN_FILE = 'data/place47_train.txt'
PLACE47_TRAIN_ROOT = '/data/place/data_large'

PLACE47_VALID_FILE = 'data/place47_valid.txt'
PLACE47_VALID_ROOT = '/data/place/val_256'

EPOCH = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

PRINT_FREQ = 30


def main():
    start_epoch = 0
    best_prec1 = 0
    log = init_logger('logs/resnet50.txt')
    model = nets.Resnet50(47).cuda()
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
                              , batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(start_epoch, EPOCH):
        # adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, PRINT_FREQ)
        # evaluate on validation set
        prec1 = validate(valid_loader, model, criterion, PRINT_FREQ)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, 'ckpts/resnet50'.format(epoch))

    # validate(valid_loader, model, criterion)


if __name__ == '__main__':
    main()
