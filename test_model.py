###
#  Author: Seya Peng
#  Test Model
###

from parts.Net.DataLoader import myDataset
from parts.Net.train import train
from parts.Net.test import test
from parts.Net.Model import ModelFactory
from parts.sViewr.sViewr import sViewr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import datetime

## import Superior documents
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from ulits import *
sys.path.pop()  # resume after using

argvList = ArgvList(sys.argv)

# to view the output with vedio
View = 'True'.lower() == argvList.getargv('view', 'False')

# load parameters
img_size = (320, 240)   # disp size
batch_size = 1 if View == True else int(argvList.getargv('batch_size', 1))
n_epochs = 1
seq_length = int(argvList.getargv('seq_length', 1))
id = argvList.getargv('device', 'cuda:0')
device = torch.device(id)
learning_rate = float(argvList.getargv('lr', 0.001))

model_type = argvList.getargv('model_type', 'cnn')
num_layers = int(argvList.getargv('num_layers', 2))
dropout = float(argvList.getargv('dropout', 0))

start = int(argvList.getargv('start', 0))
end = int(argvList.getargv('end', -1))

# load data
test_path = os.path.dirname(os.path.abspath(__file__)) + '/DataRecord/' + argvList.getargv('dataloader_path', 'train')

# load model path
load_path = os.path.dirname(os.path.abspath(__file__)) + '/models/' + argvList.getargv('load_path', 'test_model.pth')

transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.48, 0.48), (0.22, 0.22, 0.22)),
])

# dataloader
test_dataloader = DataLoader(
    myDataset(test_path, transforms_=transforms_,
    seq_length = 1, 
    in_width = img_size[0],
    in_height = img_size[1],
    start = start
    ),
    batch_size = batch_size,
)

if __name__ == "__main__":
    # create Net
    modelfactory = ModelFactory(3, 2, img_size[0], img_size[1])
    # load model
    model = modelfactory.load(load_path)
    model = model.to(device)
    
    if(model.name == 'lstm'):
        test_dataloader.dataset.seq_length = model.seq_length
    
    # init criterion
    criterion = nn.MSELoss()

    model.print_params()
    
    sviewr = None
    # view output with veido
    if(View):
        sviewr = sViewr(img_size[0], img_size[1])
    _, loss1_list, loss2_list, loss_list, tongji1_list, tongji2_list = test(test_dataloader, model, criterion, device, sviewr, start, end)
    
    # show loss plot
    loss_plot_show(loss1_list, 'Speed')
    loss_plot_show(loss2_list, 'Angle', 'orange')
    loss_plot_show(loss_list, 'All', 'red')
    show_histogram(tongji1_list, ['<0.05', '0.05~0.10', '0.10~0.15', '0.15~0.20', '0.20~0.25', '>0.25'], 'Speed')
    show_histogram(tongji2_list, ['<0.05', '0.05~0.10', '0.10~0.15', '0.15~0.20', '0.20~0.25', '0.25~0.30','0.30~0.35','>0.35'], 'Angle','red')
    if(View):
        sviewr.shutdown()




        




    
