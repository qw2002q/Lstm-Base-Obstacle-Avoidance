###
#  Author: Seya Peng
#  Dataset Loader
###

import os
import glob
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import time

## import Superior documents
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../Camera/RGBD')
from RGBD import *
sys.path.pop()  # resume after using


class myDataset(Dataset):
    def __init__(self, root, transforms_=None, seq_length = 10, in_width = 640, in_height = 240, start = 0, end = 0):
        self.seq_length = seq_length
        self.in_width = in_width
        self.in_height = in_height

        self.transform = transforms_
        self.images = sorted(glob.glob(root + "/photos/*.*"))  # height, width, channels
        grouth_truth = np.loadtxt(root + "/speedRecord_ALL", dtype = np.float32)
        self.speed = grouth_truth[:,0]
        self.angle = grouth_truth[:,1]
        self.seq = grouth_truth[:,2]

        self.rgbd = RGBD(in_width, in_height)
        self.length = int(len(self.images))
        print("Load Images: %d" % self.length)
        
        self.images_buffer = np.zeros([self.seq_length, 3, self.in_height, self.in_width * 2])
        self.disps_buffer = np.zeros([self.seq_length, 1, self.in_height, self.in_width])
        
        self.start = start
        self.end = end

        if self.length != int(self.seq.shape[0]):
            print('load data error: sequence number can not match')
            print('gt numbers: %d' % int(self.seq.shape[0]))
            exit()
    
    def __getitem__(self, i, num = 6, blockSize = 5):
        if(i == 0):
            self.images_buffer = np.zeros([self.seq_length, 3, self.in_height, self.in_width * 2])
            self.disps_buffer = np.zeros([self.seq_length, 1, self.in_height, self.in_width])
    
        # i = i + self.start
        # read image
        img = cv2.imread(self.images[i])
        img = cv2.resize(img, (self.in_width * 2, self.in_height), interpolation = cv2.INTER_CUBIC)
    
        # get disp
        disp, _ = self.rgbd.create_RGBD(img, num = num, blockSize = blockSize)
    
        img = self.transform(img)   # channels, height, width
    
        disp = disp.reshape([1, disp.shape[0], disp.shape[1]])
        disp = torch.from_numpy(disp)
    
        # Normalize
        disp = disp.float()
        disp = disp.sub_(disp.mean()).div_(disp.std())
    
        img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
        disp = disp.reshape([1, disp.shape[0], disp.shape[1], disp.shape[2]])
    
        self.images_buffer = np.delete(np.append(self.images_buffer, img, axis=0), 0, axis=0)
        self.disps_buffer = np.delete(np.append(self.disps_buffer, disp, axis=0), 0, axis=0)
    
        # read grouth truth
        speed = self.speed[i]
        angle = self.angle[i]
        gt = np.array([speed, angle])
    
        return {'image': torch.from_numpy(self.images_buffer), 'disp':torch.from_numpy(self.disps_buffer), 'gt': torch.from_numpy(gt), 'path_string': self.images[i]}

    def __len__(self):
        self.length = int(len(self.images))
        return self.length

class ImgBuffer():
    def __init__(self,transforms_=None, seq_length = 10, in_width = 640, in_height = 240):
        self.seq_length = seq_length
        self.in_width = in_width
        self.in_height = in_height
        self.initbuffer = False

        self.transform = transforms_

        self.rgbd = RGBD(in_width, in_height)
        
        self.images_buffer = np.zeros([self.seq_length, 3, self.in_height, self.in_width * 2])
        self.disps_buffer = np.zeros([self.seq_length, 1, self.in_height, self.in_width])
    
    def initBuffer(self):
        self.images_buffer = np.zeros([self.seq_length, 3, self.in_height, self.in_width * 2])
        self.disps_buffer = np.zeros([self.seq_length, 1, self.in_height, self.in_width])

    def updateBuffer(self, frame, num = 6, blockSize = 5):
        if self.initbuffer == False:
            self.initBuffer()
            self.initbuffer = True

        # read image
        img = frame
        img = cv2.resize(img, (self.in_width * 2, self.in_height), interpolation = cv2.INTER_CUBIC)
    
        # get disp
        disp, _ = self.rgbd.create_RGBD(img, num = num, blockSize = blockSize)
    
        img = self.transform(img)   # channels, height, width
    
        disp = disp.reshape([1, disp.shape[0], disp.shape[1]])
        disp = torch.from_numpy(disp)
    
        # Normalize
        disp = disp.float()
        disp = disp.sub_(disp.mean()).div_(disp.std())
    
        img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])
        disp = disp.reshape([1, disp.shape[0], disp.shape[1], disp.shape[2]])
    
        self.images_buffer = np.delete(np.append(self.images_buffer, img, axis=0), 0, axis=0)
        self.disps_buffer = np.delete(np.append(self.disps_buffer, disp, axis=0), 0, axis=0)
    
    def getBuffer(self):
        images_buffer = np.resize(self.images_buffer, [1, self.images_buffer.shape[0], 
            self.images_buffer.shape[1], self.images_buffer.shape[2], self.images_buffer.shape[3]])
        disps_buffer = np.resize(self.disps_buffer, [1, self.disps_buffer.shape[0],
            self.disps_buffer.shape[1], self.disps_buffer.shape[2], self.disps_buffer.shape[3]])
        return {'image': torch.from_numpy(images_buffer), 'disp':torch.from_numpy(disps_buffer)}