import os
import cv2
import skimage.io
import tqdm
import h5py
import zipfile
import torch
import numpy as np
import pandas as pd

TRAIN = '/home/zhaoxun/codes/Panda/_data/train_images'
MASKS = '/home/zhaoxun/codes/Panda/_data/train_label_masks'
LABEL = '/home/zhaoxun/codes/Panda/_data/train.csv'
sz = 128
N = 16

def tile(img, mask):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2, pad0-pad0 // 2], [pad1 // 2, pad1-pad1 // 2], [0, 0]], constant_values = 255)
    mask = np.pad(mask, [[pad0 // 2, pad0-pad0 // 2], [pad1 // 2, pad1-pad1 // 2], [0, 0]], constant_values = 0)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    mask = mask.reshape(mask.shape[0] // sz, sz, mask.shape[1] // sz, sz, 3)
    mask = mask.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < N:
        mask = np.pad(mask, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values = 0)
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values = 255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img': img[i], 'mask': mask[i], 'idx': i})
    return result

x_tot, x2_tot = [], []
names = [name[:-10] for name in os.listdir(MASKS)]
label = pd.read_csv(LABEL)

h5 = h5py.File("/home/zhaoxun/codes/Panda/_data/v0.h5", "w")
h5.create_dataset("img", shape = (len(names) * N, 3, sz, sz), dtype = np.uint8)
h5.create_dataset("msk", shape = (len(names) * N, sz, sz), dtype = np.uint8)
h5.create_dataset("lbl", shape = (len(names) * N, 4), dtype = np.int16)

for i, name in enumerate(tqdm.tqdm(names, desc = "iterating..", mininterval = 60)):
    img = skimage.io.MultiImage(os.path.join(TRAIN, name + '.tiff'))[-1]
    mask = skimage.io.MultiImage(os.path.join(MASKS, name + '_mask.tiff'))[-1]
    tiles = tile(img, mask)

    info = label.loc[label['image_id'] == name, ['data_provider', 'isup_grade']]
    for t in tiles:
        img, mask, idx = t['img'], t['mask'], t['idx']
        mask = mask[:, :, 0]
        if info.iloc[0,0] == 'radboud':
            for a, b in ((2, 1), (3, 2), (4, 2), (5, 2)):
                mask[mask == a] = b
        x_tot.append((img / 255.0).reshape(-1, 3).mean(0))
        x2_tot.append(((img / 255.0) ** 2).reshape(-1, 3).mean(0))
        h5["img"][i * N + t['idx']] = img.transpose((2, 0, 1))
        h5['msk'][i * N + t['idx']] = mask
        h5['lbl'][i * N + t['idx']] = np.array([i, t['idx'], info.iloc[0,0] == 'radboud', info.iloc[0,1]])

img_avr = np.array(x_tot).mean(0)
img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)

h5.create_dataset('meanstd', data = np.stack([img_avr, img_std]))

h5.close()

