import torch
import torchvision

class PoolTileNet(torch.nn.Module):
    def __init__(self, nc):
        super(PoolTileNet, self).__init__()
        net = torchvision.models.resnext50_32x4d(pretrained = True)
        self.net1 = torch.nn.Sequential(*list(net.children())[:-2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.lnr = torch.nn.Linear(2048, nc)

    def forward(self, x):
        shape = x.shape
        n = shape[1]
        x = x.view(-1,shape[2],shape[3],shape[4])
        x = self.net1(x)
        shape = x.shape
        x = x.view(-1,n,shape[1],shape[2],shape[3]).permute(0,2,1,3,4).contiguous().view(-1,shape[1],shape[2]*n,shape[3])
        x = self.avgpool(x)
        x = x.view(x.shape[:2])
        x = self.lnr(x)
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
