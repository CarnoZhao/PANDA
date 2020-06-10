DEBUG = False

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import sys
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import random
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from fastai.vision import AdaptiveConcatPool2d, Flatten
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as enet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from apex import amp
import radam
import Radam

def make_file_path():
    basepath = os.path.split(os.path.realpath(__file__))[0]
    fullpath = os.path.realpath(__file__)
    filename = os.path.split(os.path.realpath(__file__))[1]
    t = time.strftime("%b.%d_%H:%M", time.localtime()) + ".cls"
    os.system("mkdir -p _archives _outs _plots _models; cp %s %s/_archives/%s.py" % (fullpath, basepath, t))
    modelpath = os.path.join(basepath, "_models", "%s.model" % t)
    plotpath = os.path.join(basepath, "_plots", "%s.png" % t)
    outpath = os.path.join(basepath, "_outs", "%s.out" % t)
    return modelpath, plotpath, outpath, t, basepath
modelpath, plotpath, outpath, starttime, basepath = make_file_path()

def printOut(*args):
    with open(outpath, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

data_dir = '/home/zhaoxun/codes/Panda/_data'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'qishen')
# image_folder = os.path.join(data_dir, 'iafoss/train')

enet_type = 'efficientnet-b0'
fold = 0
tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 16
num_workers = 4
out_dim = 5
init_lr = 3e-4
# init_lr = 1e-3
warmup_factor = 10
warmup_epo = 1
seed = 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(seed)

n_epochs = 1 if DEBUG else 30
df_train = df_train.sample(100).reset_index(drop=True) if DEBUG else df_train
# df_train = df_train.set_index('image_id')
# files = sorted(set([p[:32] for p in os.listdir(image_folder)]))
# df_train = df_train.loc[files]
# df_train = df_train.reset_index()

device = torch.device('cuda')

skf = StratifiedKFold(5, shuffle = True, random_state = seed)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
    df_train.loc[valid_idx, 'fold'] = i
# df_train.head()

pretrained_model = {
     'efficientnet-b0': "/home/zhaoxun/codes/Panda/_data/models/efficientnet-b0-08094119.pth",
     'efficientnet-b4': '/home/zhaoxun/codes/Panda/_data/models/efficientnet-b4-6ed6700e.pth'
}

def gem(x, p=3, eps=1e-6):
    return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = torch.nn.parameter.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        # net = torchvision.models.resnext50_32x4d(pretrained = True)
        # infeature = net.fc.in_features
        # self.enet = torch.nn.Sequential(*list(net.children())[:-2])
        # self.myfc = nn.Sequential(GeM(), Flatten(), nn.Dropout(0.2), nn.Linear(infeature, out_dim))

        self.enet = enet.EfficientNet.from_name(backbone)
        state_dict = torch.load(pretrained_model[backbone])
        self.enet.load_state_dict(state_dict)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        # self.myfc2 = nn.Linear(self.enet._fc.in_features, out_dim + 1)
        # self.enet._avg_pooling = GeM()
        # self.enet._dropout = torch.nn.Dropout(0.4, inplace = False)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
        # x1 = self.myfc(x)
        # x2 = self.myfc2(x)
        # return torch.cat([x1, x2], dim = 1)

class MSECross(torch.nn.Module):
    def __init__(self):
        super(MSECross, self).__init__()
        self.loss1 = torch.nn.BCEWithLogitsLoss()
        self.loss2 = torch.nn.CrossEntropyLoss()

    def forward(self, yhat, y):
        loss1 = self.loss1(yhat[:,:5], y[:,:5])
        loss2 = self.loss2(yhat[:,5:], y[:,5].long())
        return loss1 + loss2
        

class PANDADataset(Dataset):
    def __init__(self, df, image_size, n_tiles = n_tiles, tile_mode = 0, rand = False, transform = None):
        self.df = df.reset_index(drop = True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiles = [np.array(PIL.Image.open(os.path.join(image_folder, "%s_%d.png" % (img_id, idx)))) for idx in range(n_tiles)]

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace = False)
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image = this_img)['image']
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1+image_size, w1:w1 + image_size] = this_img

        if self.transform is not None:
            images = self.transform(image = images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        # label = np.zeros(6).astype(np.float32)
        # label[:row.isup_grade] = 1.
        # label[5] = row.isup_grade
        return torch.tensor(images), torch.tensor(label)


transforms_train = albumentations.Compose([
    albumentations.Transpose(p = 0.5),
    albumentations.VerticalFlip(p = 0.5),
    albumentations.HorizontalFlip(p = 0.5),
    # albumentations.Rotate(15, border_mode = cv2.BORDER_CONSTANT, value = 255)
])
transforms_val = albumentations.Compose([])

def train_epoch(loader, optimizer):
    model.train()
    train_loss = []
    # bar = tqdm(loader, mininterval = 60)
    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        # bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(loader, get_output = False):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    acc = 0.
    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)

            ###
            # logits = logits[:,:5]
            # target = target[:,:5]
            ###

            pred = logits.sigmoid().sum(1).detach().round()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target.sum(1))
            acc += (target.sum(1) == pred).sum().cpu().numpy()
            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)
        acc = acc / len(dataset_valid) * 100
    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')
    printOut('qwk', qwk, 'qwk_k', qwk_k, 'qwk_r', qwk_r)
    if get_output:
        return LOGITS
    else:
        return val_loss, acc, qwk

train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]

dataset_train = PANDADataset(df_this , image_size, n_tiles, transform = transforms_train)
dataset_valid = PANDADataset(df_valid, image_size, n_tiles, transform = transforms_val)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, sampler = RandomSampler(dataset_train), num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size = batch_size, sampler = SequentialSampler(dataset_valid), num_workers = num_workers)

model = enetv2(enet_type, out_dim=out_dim)
# model.load_state_dict(torch.load("/home/zhaoxun/codes/Panda/_models/fold0.pth"))
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
# criterion = MSECross()
optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

# optimizer = Radam.Over9000(model.parameters(), lr = init_lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
model = torch.nn.DataParallel(model, device_ids = [0, 1])


qwk_max = 0.
for epoch in range(1, n_epochs + 1):
    printOut(time.ctime(), 'Epoch:', epoch)
    scheduler.step(epoch-1)

    train_loss = train_epoch(train_loader, optimizer)
    val_loss, acc, qwk = val_epoch(valid_loader)

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
    printOut(content)

    if qwk > qwk_max:
        printOut('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
        torch.save(model.module.state_dict(), modelpath)
        qwk_max = qwk

torch.save(model.module.state_dict(), os.path.join(modelpath.replace(".model", ".final.model")))