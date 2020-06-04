import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import fastai
from fastai.vision import *
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *

class PoolTileNet(torch.nn.Module):
    def __init__(self, nc):
        super(PoolTileNet, self).__init__()
        # net = torchvision.models.resnext50_32x4d(pretrained = True)
        net = torchvision.models.resnet34(pretrained = True)
        infeature = net.fc.in_features
        self.net1 = torch.nn.Sequential(*list(net.children())[:-2])
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(), nn.Linear(infeature * 2,512), Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,nc))

    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.net1(x)
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        return x

class MyEfficientNet(torch.nn.Module):
    def __init__(self, nc):
        super(MyEfficientNet, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        infeature = self.net._conv_head.out_channels
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(), nn.Linear(infeature * 2,512), Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,nc),MemoryEfficientSwish())

    def extract_features(self, inputs):
        x = self.net._swish(self.net._bn0(self.net._conv_stem(inputs)))
        for idx, block in enumerate(self.net._blocks):
            drop_connect_rate = self.net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.net._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate = drop_connect_rate)
        x = self.net._swish(self.net._bn1(self.net._conv_head(x)))
        return x

    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.extract_features(x)
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        x = self.head(x)
        return x
    
class FocalSmoothLoss(torch.nn.Module):
    def __init__(self, nc, sm = 0, gm = 0):
        super(FocalSmoothLoss, self).__init__()
        self.gm = gm
        self.sm = sm
        self.nc = nc

    def forward(self, Yhat, Y):
        Yhat = Yhat.cuda(); Y = Y.cuda()
        Yhat = Yhat.softmax(1)
        oneHot = torch.zeros_like(Yhat)
        oneHot = oneHot.scatter_(1, Y.data.unsqueeze(1), 1)
        oneHot = torch.clamp(oneHot, self.sm / (self.nc - 1), 1.0 - self.sm)
        pt = (oneHot * Yhat).sum(1)  + 1e-10
        logpt = pt.log()
        loss = -torch.pow(1 - pt, self.gm) * logpt
        return loss.mean()


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)

class SemiResNext(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten(),nn.Linear(2*nc,512),
                            Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        
    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1, n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        return x

class SemiResNextList(nn.Module):
    def __init__(self, arch='resnext50_32x4d_ssl', n=6, pre=True):
        super().__init__()
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', arch)
        self.enc = nn.Sequential(*list(m.children())[:-2])       
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),nn.Linear(2*nc,512),Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,n))
        # self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),nn.Dropout(0.2),nn.Linear(2 * nc,n))
        
    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.enc(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        return x

class PoolTileNetList(torch.nn.Module):
    def __init__(self, nc):
        super(PoolTileNetList, self).__init__()
        # net = torchvision.models.resnext50_32x4d(pretrained = True)
        net = torchvision.models.resnet34(pretrained = True)
        infeature = net.fc.in_features
        self.net1 = torch.nn.Sequential(*list(net.children())[:-2])
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(), nn.Linear(infeature * 2,512), Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,nc))

    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.net1(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        return x

class MyEfficientNetList(torch.nn.Module):
    def __init__(self, nc, which):
        super(MyEfficientNetList, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b' + which)
        infeature = self.net._conv_head.out_channels
        self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(), Mish(),nn.BatchNorm1d(infeature * 2), nn.Dropout(0.5),nn.Linear(infeature * 2,512), Mish(),nn.BatchNorm1d(512), nn.Dropout(0.5),nn.Linear(512,nc),MemoryEfficientSwish())
        # self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),nn.Dropout(0.2),nn.Linear(2 * infeature,nc))

    def net1(self, inputs):
        x = self.net._swish(self.net._bn0(self.net._conv_stem(inputs)))
        for idx, block in enumerate(self.net._blocks):
            drop_connect_rate = self.net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.net._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate = drop_connect_rate)
        x = self.net._swish(self.net._bn1(self.net._conv_head(x)))
        return x

    def forward(self, *x):
        shape = x[0].shape
        n = len(x)
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128
        x = self.net1(x)
        #x: bs*N x C x 4 x 4
        shape = x.shape
        #concatenate the output for tiles into a single map
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous()\
          .view(-1,shape[1],shape[2]*n,shape[3])
        #x: bs x C x N*4 x 4
        x = self.head(x)
        #x: bs x n
        return x