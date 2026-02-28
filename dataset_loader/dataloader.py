
import glob
import cv2
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import torch.nn as nn
import scipy.io as sio
from functools import partial
from edgeconstrin import *
import h5py
import torch.nn.functional as nF
from core import imresize,gaussian_blur
class DataloaderSimpleTrain(Dataset):

    def __init__(self, opt):
        self.paths = opt['paths']
        self.data_paths = []
        self.sf = opt['sf']
        self.ds = 1.0/(self.sf)
        self.gt_size = 256
        for path in self.paths:
            self.data_paths.extend(glob.glob(path + '/*.mat')) #+ glob.glob(path + '/*.jpg') + glob.glob(path + '/*.JPEG'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):

        img = sio.loadmat(self.data_paths[index])
        #img = h5py.File(self.data_paths[index])
        img = img['ref']
        img = torch.from_numpy(np.double(img))
        i, j = 500, 500
        img = img.permute(2,0,1)
        gt = img[:, i:i+self.gt_size, j:j+self.gt_size ]
        gt = gt / (torch.max(gt)-torch.min(gt))
        #SRF = sio.loadmat('SRF/128.mat')
        #SRF = torch.from_numpy(SRF['srf'])#.permute(1,0)  # .float()
        SRF = sio.loadmat('SRF/P_N_V2.mat')
        SRF = torch.from_numpy(SRF['P_20N'])
        gt1 = gt.reshape(gt.size()[0], -1)
        rgb = torch.matmul(SRF, gt1).reshape(SRF.size()[0], gt.size()[1],
                                                          gt.size()[1])
        #rgb = rgb/torch.max(rgb)-torch.min(rgb)
        down = partial(imresize, scale=0.125)
        lq = down(gaussian_blur(gt, sigma=2))
        self.gt = gt.float()
        self.lq = lq.float()
        self.rgb = rgb.float()
        return  self.gt,  self.lq , self.rgb



class DataloaderSimpleTest(Dataset):
    def __init__(self, opt):
        self.paths = opt['paths']
        self.data_paths = []
        self.sf = opt['sf']
        self.ds = 1.0/(self.sf)
        self.gt_size = 256
        for path in self.paths:
            self.data_paths.extend(glob.glob(path + '/*.mat'))
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img = sio.loadmat(self.data_paths[index])
        #img = h5py.File(self.data_paths[index])
        img = img['ref']
        img = torch.from_numpy(np.double(img))
        i, j = 500, 500
        img = img.permute(2,0,1)
        gt = img[:, i:i+self.gt_size, j:j+self.gt_size ]
        gt = gt / (torch.max(gt)-torch.min(gt))
        #SRF = sio.loadmat('SRF/128.mat')
        #SRF = torch.from_numpy(SRF['srf'])#.permute(1,0)  # .float()
        SRF = sio.loadmat('SRF/P_N_V2.mat')
        SRF = torch.from_numpy(SRF['P_20N'])
        gt1 = gt.reshape(gt.size()[0], -1)
        rgb = torch.matmul(SRF, gt1).reshape(SRF.size()[0], gt.size()[1],
                                                          gt.size()[1])
        #rgb = rgb/torch.max(rgb)-torch.min(rgb)
        down = partial(imresize, scale=0.125)
        lq = down(gaussian_blur(gt, sigma=2))
        self.gt = gt.float()
        self.lq = lq.float()
        self.rgb = rgb.float()
        return  self.gt,  self.lq , self.rgb
