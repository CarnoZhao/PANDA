import os
import sys
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

data_dir = '/home/zhaoxun/codes/Panda/_data'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'train_images')


tile_size = 256
image_size = 256
n_tiles = 36


def get_tiles(img, mode = 0):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w // 2], [0,0]], constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // tile_size,
        tile_size,
        img2.shape[1] // tile_size,
        tile_size,
        3
    )

    img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
    n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
    if len(img3) < n_tiles:
        img3 = np.pad(img3,[[0, n_tiles - len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
    return result, n_tiles_with_info >= n_tiles

for img_id in df_train.image_id:
    if os.path.exists(os.path.join(data_dir, "qishen", "%s_%d.png" % (img_id, 35))): continue
    tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
    image = skimage.io.MultiImage(tiff_file)[1]
    tiles, ok = get_tiles(image)
    for tile in tiles:
        im = PIL.Image.fromarray(tile['img'])
        im.save(os.path.join(data_dir, "qishen", "%s_%d.png" % (img_id, tile['idx'])))