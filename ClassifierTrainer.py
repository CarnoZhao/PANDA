import os
import tqdm
import time
import h5py
import random
import argparse
import traceback
import warnings
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score

root = "/home/zhaoxun/codes/Panda"
warnings.filterwarnings("ignore", category = UserWarning)
global modelpath, plotpath, outpath, starttime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_all_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--h5", help = "h5 file path", default = "/home/zhaoxun/codes/Panda/_data/v0.h5", type = str)
    parser.add_argument("-M", "--MD", help = "model file", default = "", type = str)
    parser.add_argument("-N", "--nt", help = "net type", default = "resnet34", choices = ["resnet34", "resnext50", "eff"], type = str)
    parser.add_argument("-e", "--ep", help = "number of epochs", default = 5, type = int)
    parser.add_argument("-n", "--nc", help = "number of classes", default = 6, type = int)
    # parser.add_argument("-p", "--pt", help = "pretrained", default = True, type = bool)
    parser.add_argument("-l", "--lr", help = "learning rate", default = 1e-3, type = float)
    parser.add_argument("-L", "--ls", help = "loss type", default = "cross", choices = ["cross", "focal"], type = str)
    parser.add_argument("-b", "--bs", help = "batch size", default = 32, type = int)
    # parser.add_argument("-c", "--cb", help = "call-back step size", default = 1, type = int)
    # parser.add_argument("-s", "--ss", help = "learning rate scheduler step size", default = 30, type = int)
    parser.add_argument("-w", "--wd", help = "weight decay", default = 1e-3, type = float)
    # parser.add_argument("-g", "--gp", help = "gpus", default = [0], type = list)
    parser.add_argument("-d", "--dv", help = "visible devices", default = "3", choices = list("0123"), type = str)
    parser.add_argument("-s", "--sm", help = "label smoothing", default = 0.001, type = float)
    parser.add_argument("-m", "--gm", help = "focal loss gamma", default = 2, type = int)
    parser.add_argument("-o", "--op", help = "optim method", default = "sgd", choices = ["sgd", "adam", "over"], type = str)
    parser.add_argument("-v", "--df", help = "divide factor", default = 100, type = int)
    parser.add_argument("-B", "--bg", help = "bag iters", default = 0, type = int)
    parser.add_argument("-R", "--br", help = "bag ratio", default = 0.8, type = float)
    # parser.add_argument("-S", "--sc", help = "learning rate scheduler", default = "cos", type = str)
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
import MyData
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from radam import *
from Radam import *
setup_seed(1)

class Train(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.data = MyData.PngData()

        # model
        self.load_net()

        # loss
        if kwargs['ls'] == "focal":
            self.loss = PoolTileNet.FocalSmoothLoss(kwargs['nc'], kwargs['sm'], kwargs['gm'])
        elif kwargs['ls'] == "cross":
            self.loss = torch.nn.CrossEntropyLoss()
        
        if kwargs['op'] == "sgd":
            self.opt = torch.optim.SGD
        elif kwargs['op'] == "adam":
            self.opt = RAdam
        elif kwargs['op'] == "over":
            self.opt = Over9000

    def load_net(self):
        if self.kwargs['nt'] == "resnet34":
            self.net = PoolTileNet.PoolTileNetList(self.kwargs["nc"])
        elif self.kwargs['nt'] == "resnext50":
            self.net = PoolTileNet.SemiResNextList(n = self.kwargs['nc'])
        elif self.kwargs['nt'] == "eff":
            self.net = PoolTileNet.MyEfficientNet(self.kwargs['nc'])
        self.net = self.net.cuda()
        if self.kwargs['MD']:
            dic = torch.load(self.kwargs['MD'])
            self.net.load_state_dict(dic)

    def callback_bak(self, i, loss, valloader):
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

    class callback(LearnerCallback):
        def __init__(self, ln):
            self.ln = ln
            self.losses = {"train": [], "val": []}
            

        @property
        def header(self):
            return self.ln.recorder.names

        def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
            self.losses['train'].append(float(smooth_loss))
            self.losses['val'].append(float(last_metrics[0]))
            self.write_stats([epoch, smooth_loss] + last_metrics)

        def write_stats(self, stats):
            printOut("{:3d} | tr: {:.3f} | vl: {:.3f} | kp: {:.3f}".format(*stats))

        def on_train_end(self, **kwargs):
            plt.plot(self.losses['train'], label = "train")
            plt.plot(self.losses['val'], label = "val")
            plt.legend()
            plt.savefig(plotpath)

    def evaluations_bak(self, trainloader, valloader):
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

    def evaluations(self, ln):
        pred,target = [],[]
        ln.model.eval()
        with torch.no_grad():
            for step, (x, y) in progress_bar(enumerate(ln.data.dl(DatasetType.Valid)), total=len(ln.data.dl(DatasetType.Valid))):
                p = ln.model(*x)
                pred.append(p.float().cpu())
                target.append(y.cpu())
        p = torch.argmax(torch.cat(pred, 0), 1)
        t = torch.cat(target)
        printOut("Val kappa: %.5f" % cohen_kappa_score(t, p, weights = 'quadratic'))
        printOut(confusion_matrix(t, p))

    def bag_eval(self):
        pred,target = [],[]
        dirpath = modelpath + "s"
        dl = self.data.get_data(self.kwargs['bs'], 0, self.kwargs['br'])
        for bag in range(self.kwargs['bg']):
            self.net.load_state_dict(torch.load(os.path.join(dirpath, "model.%d.pth" % bag)))
            ln = Learner(dl, self.net, loss_func = self.loss, opt_func = self.opt, metrics = [KappaScore(weights = 'quadratic')], bn_wd = False, wd = self.kwargs['wd']).to_fp16()
            ln.model.eval()
            with torch.no_grad():
                for step, (x, y) in progress_bar(enumerate(ln.data.dl(DatasetType.Valid)), total=len(ln.data.dl(DatasetType.Valid))):
                    p = ln.model(*x)
                    if len(pred) == step: 
                        pred.append(p.float().cpu())
                        target.append(y.cpu())
                    else:
                        pred[step] += p.float().cpu()
        p = torch.argmax(torch.cat(pred, 0), 1)
        t = torch.cat(target)
        printOut("Val kappa: %.5f" % cohen_kappa_score(t, p, weights = 'quadratic'))
        printOut(confusion_matrix(t, p))

    def train_bak(self):
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

    def train(self, dl = None):
        dl = self.data.toLoader(self.kwargs['bs']) if dl is None else dl
        ln = Learner(dl, self.net, loss_func = self.loss, opt_func = self.opt, metrics = [KappaScore(weights = 'quadratic')], bn_wd = False, wd = self.kwargs['wd']).to_fp16()
        ln.clip_grad = 1.0
        ln.split([self.net.head])
        ln.unfreeze()
        cb = self.callback(ln)
        ln.fit_one_cycle(self.kwargs['ep'], max_lr = self.kwargs['lr'], div_factor = self.kwargs['df'], pct_start = 0.0, wd = self.kwargs['wd'], callbacks = [cb])
        torch.save(self.net.state_dict(), modelpath)
        self.evaluations(ln)

    def bag_train(self):
        dirpath = modelpath + "s"
        os.system("mkdir -p %s" % dirpath)
        for bag in range(self.kwargs['bg']):
            dl = self.data.get_data(self.kwargs['bs'], bag, self.kwargs['br'])
            self.load_net()
            self.train(dl)
            os.system("mv %s %s" % (modelpath, os.path.join(dirpath, "model.%d.pth" % bag)))
        self.bag_eval()


if __name__ == "__main__":
    for k, v in params.items():
        printOut(k, ':', v)
    try:
        Train(**params).train() if params['bg'] == 0 else Train(**params).bag_train()
        status = "S"
    except Exception as e:
        traceback.print_exc(file = open(outpath,'a'))
        status = "F"
    for f in (modelpath, plotpath, outpath, os.path.join(basepath, "_archives", "%s.py" % starttime)):
        os.system("mv %s %s" % (f, f.replace(starttime, status + '.' + starttime)))