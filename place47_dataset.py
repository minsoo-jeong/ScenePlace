import csv
import os


dir='/data/place/filelist'
PLACE47_TRAIN_DATA_TXT='places47_val_data.txt'
NEW_PLACE47_TRAIN_DATA_TXT='place47_valid.txt'
print(os.listdir(dir))

f=open(os.path.join(dir,PLACE47_TRAIN_DATA_TXT),'r')
nf=open(os.path.join('data',NEW_PLACE47_TRAIN_DATA_TXT),'w')

l=f.readlines()
print(len(l))
print(l[:2])
print(os.path.splitext('/'.join(l[0].split(' ')[0].split('/')[2:]))[0]+'.jpg')
print(l[0].split(' ')[1].rstrip('\n'))
for row in l:
    nf.write(' '.join([os.path.splitext('/'.join(row.split(' ')[0].split('/')[2:]))[0]+'.jpg',row.split(' ')[1].rstrip('\n')])+'\n')
