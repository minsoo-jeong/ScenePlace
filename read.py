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

ct = 'categories.txt'
ct_f = open(ct, 'r')
a = ct_f.readlines()
a = list(map(lambda x: x.rstrip().split(' ')[1], a))
print(a)
gt = 'Miss_hammurabi_E09.csv'
f = open(gt, 'r')
rdr = csv.reader(f)
scene_cls = [r[1] for r in rdr]
print(scene_cls)

test_trn = trn.Compose([trn.Resize(224),
                        trn.ToTensor(),
                        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_folder = ImageFolder('Miss_Hammurabi_E09', transform=test_trn, target_transform=lambda x: int(scene_cls[x]))
test_loader = DataLoader(test_folder, batch_size=4, shuffle=False)
print(test_folder)
im = iter(test_loader).__next__()
print(im[1])
print(a[10])

criterion = torch.nn.CrossEntropyLoss()

model = nets.Resnet50(num_classes=47)
model = model.cuda()
out = model(V(im[0].cuda()))

out_prob=torch.nn.functional.softmax(out.data)
print(out_prob)
print(out.size())
score = criterion(out, im[1].cuda())
print(out)
print(score)
'''
with torch.no_grad():
    for i in test_loader:        
        print(i[1])
'''
