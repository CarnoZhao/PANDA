import os
import tqdm
import time
import h5py
import random
import argparse
import traceback
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

root = "/home/zhaoxun/codes/Panda"
warnings.filterwarnings("ignore", category = UserWarning)
global modelpath, plotpath, outpath, starttime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # set_all_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--h5", help = "h5 file path", default = "/home/zhaoxun/codes/Panda/_data/v0.h5", type = str)
    parser.add_argument("-M", "--MD", help = "model file", default = "", type = str)
    parser.add_argument("-N", "--nt", help = "net type", default = "resnet34", type = str)
    parser.add_argument("-e", "--ep", help = "number of epochs", default = 5, type = int)
    parser.add_argument("-n", "--nc", help = "number of classes", default = 6, type = int)
    # parser.add_argument("-p", "--pt", help = "pretrained", default = True, type = bool)
    parser.add_argument("-l", "--lr", help = "learning rate", default = 1e-3, type = float)
    parser.add_argument("-L", "--ls", help = "loss type", default = "cross", type = str)
    parser.add_argument("-b", "--bs", help = "batch size", default = 32, type = int)
    parser.add_argument("-c", "--cb", help = "call-back step size", default = 1, type = int)
    # parser.add_argument("-s", "--ss", help = "learning rate scheduler step size", default = 30, type = int)
    parser.add_argument("-w", "--wd", help = "weight decay", default = 1e-3, type = float)
    parser.add_argument("-g", "--gp", help = "gpus", default = [0], type = list)
    parser.add_argument("-d", "--dv", help = "visible devices", default = "3", type = str)
    parser.add_argument("-s", "--sm", help = "label smoothing", default = 0.001, type = float)
    parser.add_argument("-m", "--gm", help = "focal loss gamma", default = 2, type = int)
    parser.add_argument("-o", "--op", help = "optim method", default = "sgd", type = str)
    parser.add_argument("-S", "--sc", help = "learning rate scheduler", default = "cos", type = str)
    return vars(parser.parse_args())

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

def printOut(*args):
    with open(outpath, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

modelpath, plotpath, outpath, starttime, basepath = make_file_path()
params = parse_args()
printOut("using GPU: " + params['dv'])
os.environ["CUDA_VISIBLE_DEVICES"] = params.pop("dv")
import torch
import torchvision
import PoolTileNet
from fastai import *
from fastai.vision import *
setup_seed(1)

class Data(object):
    def __init__(self, h5):
        h = h5py.File(h5, 'r')
        self.img = h['img']
        self.msk = h['msk']
        self.lbl = h['lbl']
        self.meanstd = h['meanstd']
        length = self.lbl.shape[0]
        lennames = len(set(self.lbl[:,0]))
        self.trainidx = np.arange(round(0.7 * lennames))
        self.validx = np.arange(round(0.7 * lennames), lennames)

    def toLoader(self, batch_size):
        trainDataset = self.Dataset(self.img, self.msk, self.lbl, self.meanstd, self.trainidx)
        valDataset = self.Dataset(self.img, self.msk, self.lbl, self.meanstd, self.validx)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size = 1, shuffle = False, num_workers = 4)
        # return ImageDataBunch(trainLoader, valLoader)
        return trainLoader, valLoader

    # torchvision.transforms.RandomHorizontalFlip(p=0.5)
    # torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, img, msk, lbl, meanstd, idx):
            self.img = img
            self.msk = msk
            self.lbl = lbl[:,3].astype(np.int)
            self.mean = meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
            self.std = meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            self.idx = idx

        def __getitem__(self, i):
            x = (self.img[self.idx[i] * 16:(self.idx[i] + 1) * 16] / 255.).astype(np.float32)
            x = (x - self.mean) / self.std
            y = self.lbl[self.idx[i] * 16]
            return x.astype(np.float32), y

        def __len__(self):
            return len(self.idx)

class Train(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data = Data(kwargs['h5'])

        # model
        if kwargs['nt'] == "resnet34":
            self.net = PoolTileNet.PoolTileNet(kwargs["nc"])
        elif kwargs['nt'] == "resnext50":
            self.net = PoolTileNet.SemiResNext(n = kwargs['nc'])
        self.net = self.net.cuda()
        if kwargs['MD']:
            dic = torch.load(kwargs['MD'])
            self.net.load_state_dict(dic)

        # loss
        if kwargs['ls'] == "focal":
            self.loss = PoolTileNet.FocalSmoothLoss(kwargs['nc'], kwargs['sm'], kwargs['gm'])
        elif kwargs['ls'] == "cross":
            self.loss = torch.nn.CrossEntropyLoss()
        
        # opts
        convlinearparams = []
        for m in self.net.modules():
            if type(m) is torch.nn.Conv2d or type(m) is torch.nn.Linear:
                convlinearparams.extend(m.parameters())
        otherparams = [p for p in self.net.parameters() if id(p) not in [id(pp) for pp in convlinearparams]]
        if kwargs['op'] == "sgd":
            self.opt = torch.optim.SGD([
                {"params": convlinearparams, "weight_decay": kwargs['wd']},
                {"params": otherparams}
            ], lr = kwargs['lr'], momentum = 0.9, nesterov = True)
        elif kwargs['op'] == "adam":
            self.opt = torch.optim.Adam([
                {"params": convlinearparams, "weight_decay": kwargs['wd']},
                {"params": otherparams}
            ], lr = kwargs['lr'])
        
        # scheduler
        if kwargs['sc'] == "plat":
            self.sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min", patience = 2)
        elif kwargs['sc'] == "cos":
            self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, kwargs['ep'])


        self.losses = {"train": [], "val": []}

    def callback(self, i, loss, valloader):
        self.losses["train"].append(loss)
        self.net.eval()
        with torch.no_grad():
            valloss = 0; valcnt = 0
            for x, y in tqdm.tqdm(valloader, desc = "Validating...", leave = False, mininterval = 60):
                x = x.cuda(); y = y.cuda()
                yhat = self.net(x)
                cost = self.loss(yhat, y.long())
                valloss += cost.item() * len(x); valcnt += len(x)
            self.losses["val"].append(valloss / valcnt)
        printOut("{:3d} | tr: {:.3f} | vl: {:.3f}".format(i, loss, valloss / valcnt))
        return valloss / valcnt

    def evaluations(self, trainloader, valloader):
        K = self.kwargs['nc']
        with torch.no_grad():
            self.net.eval()
            for i, loader in enumerate((trainloader, valloader)):
                Y = np.zeros(len(loader.dataset), dtype = np.int); Yhat = np.zeros(len(loader.dataset), dtype = np.int); idx = 0
                for x, y in tqdm.tqdm(loader, desc = "Evaluating...", leave = False, mininterval = 60):
                    x = x.cuda(); y = y.numpy()
                    yhat = self.net(x).argmax(1).cpu().data.numpy()
                    Y[idx:idx + len(y)] = y; Yhat[idx:idx + len(y)] = yhat; idx += len(y)
                printOut(["Train", "Val"][i] + " kappa: %.4f" % (metrics.cohen_kappa_score(Y, Yhat, weights = "quadratic")))
                printOut(["Train", "Val"][i] + ":\n" + str(metrics.confusion_matrix(Y, Yhat)))
                printOut("\n~~~~~\n")
        plt.plot(self.losses['train'], label = "train")
        plt.plot(self.losses['val'], label = "val")
        plt.legend()
        plt.savefig(plotpath)

    def train(self):
        trainloader, valloader = self.data.toLoader(self.kwargs['bs'])
        for i in tqdm.tqdm(range(1, self.kwargs['ep'] + 1), desc = "Iterating...", mininterval = 60):
            self.net.train()    
            loss = 0; cnt = 0
            for x, y in tqdm.tqdm(trainloader, desc = "Training...", leave = False, mininterval = 60):
                x = x.cuda(); y = y.cuda()
                yhat = self.net(x)
                cost = self.loss(yhat, y.long())
                loss += cost.item() * len(x); cnt += len(x)
                self.opt.zero_grad()
                cost.backward()
                self.opt.step()
            valloss = self.callback(i, loss / cnt, valloader)
            self.sch.step(valloss)
        torch.save(self.net.state_dict(), modelpath)
        self.evaluations(trainloader, valloader)

if __name__ == "__main__":
    for k, v in params.items():
        printOut(k, ':', v)
    try:
        Train(**params).train()
        status = "S"
    except Exception as e:
        traceback.print_exc(file = open(outpath,'a'))
        status = "F"
    for f in (modelpath, plotpath, outpath, os.path.join(basepath, "_archives", "%s.py" % starttime)):
        os.system("mv %s %s" % (f, f.replace(starttime, status + '.' + starttime)))