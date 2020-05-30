import os
import tqdm
import time
import h5py
import torch
import random
import argparse
import traceback
import torchvision
import UNet
import warnings
import numpy as np
import matplotlib.pyplot as plt

root = "/home/zhaoxun/codes/Panda"
warnings.filterwarnings("ignore", category = UserWarning)
global modelpath, plotpath, outpath, starttime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--h5", help = "h5 file path", default = "/home/zhaoxun/codes/Panda/_data/v0.h5", type = str)
    parser.add_argument("-M", "--MD", help = "model file", default = "", type = str)
    parser.add_argument("-e", "--ep", help = "number of epochs", default = 5, type = int)
    parser.add_argument("-n", "--nc", help = "number of classes", default = 3, type = int)
    # parser.add_argument("-p", "--pt", help = "pretrained", default = True, type = bool)
    parser.add_argument("-l", "--lr", help = "learning rate", default = 1e-3, type = float)
    parser.add_argument("-b", "--bs", help = "batch size", default = 128, type = int)
    parser.add_argument("-c", "--cb", help = "call-back step size", default = 1, type = int)
    # parser.add_argument("-s", "--ss", help = "learning rate scheduler step size", default = 30, type = int)
    parser.add_argument("-w", "--wd", help = "weight decay", default = 1e-3, type = float)
    parser.add_argument("-W", "--wt", help = "weight of loss", default = [1, 1, 5], type = list)
    parser.add_argument("-g", "--gp", help = "gpus", default = [0], type = list)
    parser.add_argument("-d", "--dv", help = "visible devices", default = "0", type = str)
    parser.add_argument("-s", "--sm", help = "label smoothing", default = 0.001, type = float)
    parser.add_argument("-m", "--gm", help = "focal loss gamma", default = 2, type = int)
    return vars(parser.parse_args())

def make_file_path():
    basepath = os.path.split(os.path.realpath(__file__))[0]
    fullpath = os.path.realpath(__file__)
    filename = os.path.split(os.path.realpath(__file__))[1]
    t = time.strftime("%b.%d_%H:%M", time.localtime())
    os.system("mkdir -p _archives _outs _plots _models; cp %s %s/_archives/%s.py" % (fullpath, basepath, t))
    modelpath = os.path.join(basepath, "_models", "%s.model" % t)
    plotpath = os.path.join(basepath, "_plots", "%s.png" % t)
    outpath = os.path.join(basepath, "_outs", "%s.out" % t)
    return modelpath, plotpath, outpath, t, basepath

def printOut(*args):
    with open(outpath, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

class Data(object):
    def __init__(self, h5):
        h = h5py.File(h5, 'r')
        self.img = h['img']
        self.msk = h['msk']
        self.lbl = h['lbl']
        self.meanstd = h['meanstd']
        length = self.lbl.shape[0]
        lennames = len(set(self.lbl[:,0]))
        self.trainidx = np.arange(length // lennames * round(0.7 * lennames))
        self.validx = np.arange(length // lennames * round(0.7 * lennames), length)

    def toLoader(self, batch_size):
        trainDataset = self.Dataset(self.img, self.msk, self.meanstd, self.trainidx)
        valDataset = self.Dataset(self.img, self.msk, self.meanstd, self.validx)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = batch_size, shuffle = True, num_workers = 4)
        valLoader = torch.utils.data.DataLoader(valDataset, batch_size = 1, shuffle = False, num_workers = 4)
        return trainLoader, valLoader

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, img, msk, meanstd, idx):
            self.img = img
            self.msk = msk
            self.idx = idx
            self.mean = meanstd[0][:, np.newaxis, np.newaxis]
            self.std = meanstd[1][:, np.newaxis, np.newaxis]

        def __getitem__(self, i):
            x = self.img[self.idx[i]] / 255.
            x = (x - self.mean) / self.std
            y = self.msk[self.idx[i]]
            return x.astype(np.float32), y

        def __len__(self):
            return len(self.idx)

class Train(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data = Data(kwargs['h5'])
        self.net = UNet.UNet(3, kwargs["nc"])
        self.net = self.net.cuda()
        if kwargs['MD']:
            dic = torch.load(kwargs['MD'])
            self.net.load_state_dict(dic)
        self.loss = UNet.FocalSmoothLoss(kwargs['nc'], kwargs['sm'], kwargs['gm'], kwargs['wt'])
        self.opt = torch.optim.SGD(self.net.parameters(), lr = self.kwargs['lr'], momentum = 0.9, nesterov = True, weight_decay = kwargs['wd'])
        self.sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, "min", patience = 3)
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
                IOUs = [0 for _ in range(K - 1)]; cnt = 0
                for x, y in tqdm.tqdm(loader, desc = "Evaluating...", leave = False, mininterval = 60):
                    x = x.cuda(); y = y.cuda()
                    yhat = self.net(x).argmax(1)
                    tmp = 0
                    for k in range(1, K):
                        union = torch.sum(torch.bitwise_or(yhat == k, y == k), dim = (1, 2)).cpu() * 1.
                        intersect = torch.sum(torch.bitwise_and(yhat == k, y == k), dim = (1, 2)).cpu() * 1.
                        iou = torch.sum(intersect / (union + 1e-10))
                        IOUs[k - 1] += iou
                    cnt += len(x)
                printOut(["Train", "Val"][i] + " ".join([" IOU%d: %.4f" % (k + 1, IOUs[k] / cnt) for k in range(len(IOUs))]) + " IOU: %.4f" % (sum(IOUs) / len(IOUs) / cnt))
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
    setup_seed(1)
    modelpath, plotpath, outpath, starttime, basepath = make_file_path()
    params = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = params.pop("dv")
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