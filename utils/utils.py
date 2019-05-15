import torch
import shutil
import os
import logging
from .TlsSMTPHandler import TlsSMTPHandler
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '-latest.pth.tar')
    if is_best:
        #torch.save(state, filename + '_best.pth.tar')
        shutil.copyfile(filename + '-latest.pth.tar', filename + '-best.pth.tar')




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    print(optimizer.param_groups[0].keys())
    lr = float(optimizer.param_groups[0]['lr']) * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_logger(log_path):
    os.environ['TZ'] = 'Asia/Seoul'
    time.tzset()
    base_dir = os.path.abspath(os.path.dirname(log_path))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    log = logging.getLogger('my')
    log.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('[ %(asctime)s ] %(message)s')
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    smpt_handler = TlsSMTPHandler(("smtp.naver.com", 587), 'jms8167@naver.com', ['jms8167@gmail.com'], 'Error found!',
                                  ('jms8167', 's011435a!'))
    smpt_handler.setLevel(logging.ERROR)
    smpt_handler.setFormatter(formatter)

    log.addHandler(stream_handler)
    log.addHandler(file_handler)
    log.addHandler(smpt_handler)

    return log