###
#  Author: Seya Peng
#  Train Model
###

from parts.Net.DataLoader import myDataset
from parts.Net.train import train
from parts.Net.test import test
from parts.Net.Model import ModelFactory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

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
batch_size = int(argvList.getargv('batch_size', 16))
n_epochs = int(argvList.getargv('n_epochs', 70))
seq_length = int(argvList.getargv('seq_length', 1))
pretrain_model = argvList.getargv('pretrain_model', 'False')
id = argvList.getargv('device', 'cuda:0')
device = torch.device(id)
learning_rate = float(argvList.getargv('lr', 0.001))
__init_weight = True  # don't change here

model_type = argvList.getargv('model_type', 'lstm')
num_layers = int(argvList.getargv('num_layers', 2))
dropout = float(argvList.getargv('dropout', 0))

# load data
train_path = os.path.dirname(os.path.abspath(__file__)) + '/DataRecord/' + argvList.getargv('dataloader_path', 'train')

# load model path
load_path = argvList.getargv('load_path', 'None')

# dataloader
transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.48, 0.48), (0.22, 0.22, 0.22)),
])

train_dataloader = DataLoader(
    myDataset(train_path, transforms_=transforms_,
    seq_length = seq_length, 
    in_width = img_size[0],
    in_height = img_size[1]
    ),
    batch_size = batch_size,
)

if __name__ == "__main__":
    # create Net
    modelfactory = ModelFactory(3, 2, img_size[0], img_size[1])
    
    # load model
    if(not load_path == 'None'):
        load_path = os.path.dirname(os.path.abspath(__file__)) + '/models/' + load_path
        model = modelfactory.load(load_path)
        __init_weight = False
    else:
        model = modelfactory.get_model(model_type, seq_length, num_layers, dropout)
    # pretrain model
    if(pretrain_model.lower() == 'true'):
        model.loss = float('inf')
    model.print_params()
    print('---------------')
    print('n_epochs: %d\nlr: %s\nBatch size: %d\ndevice: %s' % (n_epochs, learning_rate, batch_size, id))
    
    if(model.name == 'cnn'):
        train_dataloader.dataset.seq_length = 1
    else:
        train_dataloader.dataset.seq_length = model.seq_length
    
    # init criterion and optimize
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), betas = (0.5, 0.999), lr = learning_rate)

    # train model
    train_loss = train(train_dataloader, model, criterion, optimizer, n_epochs, 
        lr = learning_rate, 
        device = device,
        save_flag = True,
        save_name = argvList.getargv('save_path', 'trained_model.pth'),
        init_weight = __init_weight)
        
    # test(train_dataloader, model, criterion, device, View)
    
    # show loss plot
    loss_plot_show(train_loss)
