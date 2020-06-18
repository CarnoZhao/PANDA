DEBUG = False
gpus = "0,1"
enet_type = 'efficientnet-b0'
fold = 0
tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 16
num_workers = 4
out_dim = 6
init_lr = 6e-4
warmup_factor = 10
warmup_epo = 2
seed = 0
n_epochs = 1 if DEBUG else 30 + warmup_epo
weight = 1.2
isgem = True

import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
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
from sklearn.metrics import cohen_kappa_score, confusion_matrix
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(seed)

df_train = df_train.sample(100).reset_index(drop=True) if DEBUG else df_train

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
    return torch.nn.functional.avg_pool2d(x.clamp(min = eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class MyBCELoss(torch.nn.Module):
    def __init__(self):
        super(MyBCELoss, self).__init__()

    def forward(self, input, target, weight):
        return F.binary_cross_entropy_with_logits(input, target, weight.unsqueeze(1), pos_weight = None, reduction = 'mean')

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing = 0.0, dim = -1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim = self.dim))

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
        self.enet = enet.EfficientNet.from_name(backbone)
        state_dict = torch.load(pretrained_model[backbone])
        self.enet.load_state_dict(state_dict)
        self.myfc = nn.Linear(self.enet._fc.in_features * (1 if isgem else 2), out_dim)
        self.enet._avg_pooling = GeM() if isgem else AdaptiveConcatPool2d() 
        # self.enet._dropout = torch.nn.Dropout(0.4, inplace = False)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
        
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

        n_row_tiles = int(np.sqrt(self.n_tiles))
        idxes = list(range(self.n_tiles))
        if self.rand:
            idxes = np.random.choice(idxes, len(idxes), replace = False)
            # for i in range(n_row_tiles):
            #     idxes[i * n_row_tiles: (i + 1) * n_row_tiles] = np.random.choice(idxes[i * n_row_tiles: (i + 1) * n_row_tiles], n_row_tiles, replace = False)

        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3), dtype = np.uint8)
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

        # label = np.zeros(5).astype(np.float32)
        # label[:row.isup_grade] = 1.
        return torch.tensor(images), row.isup_grade, 1 if row.data_provider == "karolinska" else weight


transforms_train = albumentations.Compose([
    albumentations.Transpose(p = 0.5),
    albumentations.VerticalFlip(p = 0.5),
    albumentations.HorizontalFlip(p = 0.5),
    albumentations.RandomBrightnessContrast(p = 0.5),
    albumentations.Rotate(20, border_mode = cv2.BORDER_CONSTANT, value = 0)
])
transforms_val = albumentations.Compose([])

def train_epoch(loader, optimizer):
    model.train()
    train_loss = []
    # bar = tqdm(loader, mininterval = 60)
    for (data, target, data_provider) in loader:
        data, target, data_provider = data.to(device), target.to(device), data_provider.to(device)
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


def val_epoch(loader, tta = False):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    acc = 0.
    with torch.no_grad():
        for (data, target, data_provider) in loader:
            data, target, data_provider = data.to(device), target.to(device), data_provider.to(device)
            if tta:
                logits = torch.stack([model(datai) for datai in [
                        data,
                        data.flip(-1), data.flip(-2), data.flip(-1, -2), 
                        data.transpose(-1,-2), data.transpose(-1,-2).flip(-1), 
                        data.transpose(-1,-2).flip(-2), data.transpose(-1,-2).flip(-1, -2)
                    ]
                ]).mean(0)
            else:
                logits = model(data)
            loss = criterion(logits, target)
            pred = logits.argmax(1).detach()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target)
            acc += (target == pred).sum().cpu().numpy()
            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)
        acc = acc / len(dataset_valid) * 100
    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')
    printOut('qwk %.5f' % qwk, 'qwk_k %.5f' % qwk_k, 'qwk_r %.5f' % qwk_r)
    if tta:
        printOut(confusion_matrix(TARGETS, PREDS))
        printOut(confusion_matrix(
            df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values,
            PREDS[df_valid['data_provider'] == 'karolinska']
        ))
        printOut(confusion_matrix(
            df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values,
            PREDS[df_valid['data_provider'] == 'radboud']
        ))
    return val_loss, acc, qwk

train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]

dataset_train = PANDADataset(df_this , image_size, n_tiles, rand = True, transform = transforms_train)
dataset_valid = PANDADataset(df_valid, image_size, n_tiles, transform = transforms_val)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, sampler = RandomSampler(dataset_train), num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size = batch_size, sampler = SequentialSampler(dataset_valid), num_workers = num_workers)

model = enetv2(enet_type, out_dim = out_dim)
model = model.to(device)

criterion = LabelSmoothingLoss(out_dim, smoothing = 0.1)
# criterion = nn.CrossEntropyLoss()
# criterion = MyBCELoss()
optimizer = optim.Adam(model.parameters(), lr = init_lr / warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier = warmup_factor, total_epoch = warmup_epo, after_scheduler = scheduler_cosine)

# optimizer = Radam.Over9000(model.parameters(), lr = init_lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
model = torch.nn.DataParallel(model, device_ids = list(range(len(gpus.split(",")))))


qwk_max = 0.
for epoch in range(1, n_epochs + 1):
    printOut(time.ctime(), 'Epoch:', epoch)
    scheduler.step(epoch - 1)

    train_loss = train_epoch(train_loader, optimizer)
    val_loss, acc, qwk = val_epoch(valid_loader, epoch == n_epochs)

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
    printOut(content)

    if qwk > qwk_max:
        printOut('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
        torch.save(model.module.state_dict(), modelpath)
        qwk_max = qwk

torch.save(model.module.state_dict(), os.path.join(modelpath.replace(".model", ".final.model")))