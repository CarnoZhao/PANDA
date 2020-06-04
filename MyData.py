import torch
import torchvision
import h5py
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
from fastai import *
from fastai.vision import *
from sklearn.model_selection import StratifiedKFold

# TRAIN = "/home/zhaoxun/codes/Panda/_data/iafoss/train"
# LABELS = '/home/zhaoxun/codes/Panda/_data/train.csv'


class PngData(object):
    def __init__(self, TRAIN = "/home/zhaoxun/codes/Panda/_data/iafoss/train", LABELS = '/home/zhaoxun/codes/Panda/_data/train.csv', N = 12, size = 128):
        self.TRAIN = TRAIN
        self.mean = torch.tensor([1.0 - 0.90949707, 1.0 - 0.8188697, 1.0 - 0.87795304])[..., None, None]
        self.std = torch.tensor([0.36357649, 0.49984502, 0.40477625])[..., None, None]
        self.sz = size
        self.N = N

        nfolds = 5
        df = pd.read_csv(LABELS).set_index('image_id')
        files = sorted(set([p[:32] for p in os.listdir(TRAIN)]))
        df = df.loc[files]
        df = df.reset_index()
        splits = StratifiedKFold(n_splits = nfolds, random_state =1, shuffle =True)
        splits = list(splits.split(df, df.isup_grade))
        folds_splits = np.zeros(len(df)).astype(np.int)
        for i in range(nfolds):
            folds_splits[splits[i][1]] = i
        df['split'] = folds_splits
        self.df = df

    @classmethod
    def open_image(cls, fn, div = True, convert_mode = 'RGB', imcls = Image, after_open = None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            x = PIL.Image.open(fn).convert(convert_mode)
        if after_open:
            x = after_open(x)
        x = pil2tensor(x, np.float32)
        if div: x.div_(255)
        return imcls(1.0 - x)

    @property
    def MyImage(self):
        outter = self
        class _MyImage(ItemBase):
            def __init__(self, imgs):
                self.obj = (imgs)
                self.data = [(img.data - outter.mean) / outter.std for img in imgs]

            def __repr__(self): 
                return f'{self.__class__.__name__} {[img.shape for img in self.obj]}'

            def apply_tfms(self, tfms, *args, **kwargs):
                for i in range(len(self.obj)):
                    self.obj[i] = self.obj[i].apply_tfms(tfms, *args, **kwargs)
                    self.data[i] = (self.obj[i].data - outter.mean) / outter.std
                return self

            def to_one(self):
                img = torch.stack(self.data, 1)
                img = (img
                    .view(3, -1, outter.N, outter.sz, outter.sz)
                    .permute(0, 1, 3, 2, 4)
                    .contiguous()
                    .view(3, -1, outter.sz, outter.N))
                return Image(1.0 - (outter.mean + img * outter.std))
        return _MyImage

    @property
    def MyImageItemList(self):
        outter = self
        class _MyImageItemList(ImageList):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __len__(self): 
                return len(self.items) or 1

            def get(self, i):
                fn = Path(self.items[i])
                fnames = [Path("%s_%d.png" % (fn, i)) for i in range(outter.N)]
                imgs = [outter.open_image(fname, convert_mode = self.convert_mode, after_open = self.after_open) for fname in fnames]
                return outter.MyImage(imgs)

            def reconstruct(self, t):
                return outter.MyImage([outter.mean + ti * outter.std for ti in t])

            def show_xys(self, xs, ys, figsize = (300, 50), **kwargs):
                rows = min(len(xs), 8)
                fig, axs = plt.subplots(rows, 1, figsize = figsize)
                for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
                    xs[i].to_one().show(ax = ax, y = ys[i], **kwargs)
                plt.tight_layout()
        return _MyImageItemList

    def MImage_collate(self, batch) -> Tensor:
        result = torch.utils.data.dataloader.default_collate(to_data(batch))
        if isinstance(result[0], list):
            result = [torch.stack(result[0], 1), result[1]]
        return result

    def get_data(self, bs, bg = None, br = 0.8, subsample = 1):
        trainidx = self.df.index[self.df.split != 0].tolist()
        if bg is not None: trainidx = np.random.choice(trainidx, round(len(trainidx) * br), replace = False)
        validx = self.df.index[self.df.split == 0].tolist()
        if subsample != 1:
            trainidx = np.random.choice(trainidx, round(len(trainidx) * subsample), replace = False)
            validx = np.random.choice(validx, round(len(validx) * subsample), replace = False)
        return (self.MyImageItemList.from_df(self.df, path = '/', folder = self.TRAIN, cols = 'image_id')
                .split_by_idxs(trainidx, validx)
                .label_from_df(cols = ['isup_grade'])
                .transform(get_transforms(flip_vert = True, max_rotate = 15), size = self.sz, padding_mode = 'zeros')
                .databunch(bs = bs, num_workers = 4))


class H5Data(object):
    def __init__(self, h5, ratio = 0.8):
        h = h5py.File(h5, 'r')
        self.img = h['img']
        self.msk = h['msk']
        self.lbl = h['lbl']
        self.meanstd = h['meanstd']
        length = self.lbl.shape[0]
        lennames = len(set(self.lbl[:,0]))
        self.trainidx = np.arange(round(ratio * lennames))
        self.validx = np.arange(round(ratio * lennames), lennames)
        self.trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15)
        ])

    def toLoader(self, batch_size):
        trainDataset = self.Dataset(self.img, self.msk, self.lbl, self.meanstd, self.trainidx, self.trans)
        valDataset = self.Dataset(self.img, self.msk, self.lbl, self.meanstd, self.validx)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size = 1, shuffle = False, num_workers = 4)
        return ImageDataBunch(trainLoader, valLoader, device = "cuda")

    def toBagLoader(self, batch_size, bag_ratio = 0.8):
        bagidx = np.random.choice(self.trainidx, round(bag_ratio * len(self.trainidx)))
        trainDataset = self.Dataset(self.img, self.msk, self.lbl, self.meanstd, bagidx, self.trans)
        valDataset = self.Dataset(self.img, self.msk, self.lbl, self.meanstd, self.validx)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size = 1, shuffle = False, num_workers = 4)
        return ImageDataBunch(trainLoader, valLoader, device = "cuda")

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, img, msk, lbl, meanstd, idx, trans = None):
            self.img = img
            self.msk = msk
            self.lbl = lbl[:,3].astype(np.int)
            self.mean = meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
            self.std = meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            self.idx = idx
            self.trans = trans

        def __getitem__(self, i):
            x = 255 - self.img[self.idx[i] * 16:(self.idx[i] + 1) * 16]
            y = self.lbl[self.idx[i] * 16]
            if self.trans:
                for j in range(len(x)):
                    tmp = PIL.Image.fromarray(x[j].transpose(1, 2, 0))
                    x[j] = np.array(self.trans(tmp)).transpose(2, 0, 1)
            x = (x / 255. - 1 + self.mean) / np.sqrt(self.std)
            return x.astype(np.float32), y

        def __len__(self):
            return len(self.idx)
