import cv2
import h5py
import torch
import torchvision
import Panda.UNet as u
import numpy as np

def maskGenerate():
    net = u.UNet(3, 3)
    net = net.cuda()
    net.load_state_dict(torch.load("/home/zhaoxun/codes/Panda/_models/S.May.30_08:42.model"))
    h5 = h5py.File("/home/zhaoxun/codes/Panda/_data/v0.h5", 'r')
    idx = 1000
    img = h5['img'][idx]
    #img.shape (3, 128, 128)
    msk = h5['msk'][idx]
    #msk.shape (128, 128)
    img = np.concatenate([np.concatenate([h5['img'][i * 4 + j] for j in range(4)], axis = 1) for i in range(4)], axis = 2)
    msk = np.concatenate([np.concatenate([h5['msk'][i * 4 + j] for j in range(4)], axis = 0) for i in range(4)], axis = 1)

    mean = h5['meanstd'][0][:, np.newaxis, np.newaxis]
    std = h5['meanstd'][1][:, np.newaxis, np.newaxis]
    normimg = (img / 255. - mean) / std
    newmsk = net(torch.Tensor(normimg).unsqueeze(0).cuda())
    newmsk = newmsk[0].argmax(0).cpu().data.numpy()

    writeimg = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR) 
    msk1 = np.zeros((*msk.shape, 3))
    msk1[:, :, 1] = (msk == 1) * 255
    msk1[:, :, 2] = (msk == 2) * 255
    msk2 = np.zeros((*msk.shape, 3))
    msk2[:, :, 1] = (newmsk == 1) * 255
    msk2[:, :, 2] = (newmsk == 2) * 255
    cv2.imwrite("/home/zhaoxun/codes/Panda/_plots/test.png", np.concatenate([writeimg, msk1, msk2], axis = 0))


    