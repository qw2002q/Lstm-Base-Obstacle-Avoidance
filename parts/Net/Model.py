###
#  Author: Seya Peng
#  Network of LSTM and CNN
###

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import time
import datetime

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size = 3, stride = 1, padding = 1):
        super(CNN, self).__init__()
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernal_size,stride,padding,bias=False),   #cut a half
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)        

class ModelFactory:
    def __init__(self, in_channels, out_channels, in_width = 320, in_height = 240):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_width = in_width
        self.in_height = in_height
        
    def get_model(self, model_name, seq_length = 1, num_layers = 1, dropout = 0, device = torch.device('cuda:0'), test_time = False):
        if(model_name.lower() == 'lstm'):
            return ModelBaseLSTM(self.in_channels, self.out_channels, self.in_width, self.in_height, seq_length = seq_length, num_layers = num_layers, dropout = dropout, device = torch.device('cuda:0'), test_time = test_time)
        elif(model_name.lower() == 'cnn'):
            return ModelBaseCNN(self.in_channels, self.out_channels, self.in_width, self.in_height)
        else:
            print("Model Name Error")
            exit()
            
    def load(self, path):
        model_param = torch.load(path)
        if(model_param['name'] == 'lstm'):
            model = ModelBaseLSTM(self.in_channels, self.out_channels, self.in_width, self.in_height)
        elif(model_param['name'] == 'cnn'):
            model = ModelBaseCNN(self.in_channels, self.out_channels, self.in_width, self.in_height)
        else:
            print("LOAD MODEL ERROR")
            exit()
        
        model.load(path)
        
        print("MODEL LOADED")
        return model

class preCNN(nn.Module):
    def __init__(self, in_channels, in_width = 320, in_height = 240):
        super(preCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels,4, [4,4], [2,4], 1) # out = (1, height /2, width / 4)
      
        rate_x = in_width // 8
        rate_y = in_height // 6
        self.dispcnn = nn.Conv2d(1, 4, 4, 2, 1)
        self.cnn2 = CNN(8, 512, 3, [rate_y, rate_x], 1) # out = (512, 3, 4)
        
    def forward(self, rgb, disp):
        out = self.cnn1(rgb)
        disp = self.dispcnn(disp)

        out = torch.cat((out,disp), 1)
        
        out = self.cnn2(out)  

        
        return out

class ModelBaseLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, in_width = 320, in_height = 240, seq_length = 1, num_layers = 1, dropout = 0, device = torch.device('cuda:0'), test_time = False):
        super(ModelBaseLSTM, self).__init__()
        # parameters
        self.width = in_width
        self.height = in_height
        self.dropout = dropout
        self.num_layers = num_layers
        self.name = 'lstm'
        self.seq_length = seq_length
        self.device = device
        self.loss = float('inf')
       
        # self.ht_is_init = False
       
        self.precnn = preCNN(in_channels, in_width, in_height)
        self.lstm = nn.LSTM(6144, 128, num_layers = self.num_layers, batch_first = True, dropout = self.dropout) 
        
        # self.fc = nn.Linear(128, out_channels)
        self.fcn = nn.Conv2d(128, out_channels, 1, 1, 0)
    
    def forward(self, rgb, disp):
        batch = rgb.shape[0]
        seq = rgb.shape[1]

        # if self.ht_is_init == False:
        #     self.ht = torch.zeros(self.num_layers, batch, 128).to(self.device)
        #     self.ct = torch.zeros(self.num_layers, batch, 128).to(self.device)
        #     self.ht_is_init = True

        rgb = rgb.reshape([batch * seq, rgb.shape[2], rgb.shape[3], rgb.shape[4]])   # transform (batch, seq, channel, width, height) into (*, channel, width, height)
        disp = disp.reshape([batch * seq, disp.shape[2], disp.shape[3], disp.shape[4]])

        out = self.precnn(rgb, disp)

        out = out.reshape([batch, seq, 6144]) # transform into (batch, seq, feature)
        
        out, (ht, ct) = self.lstm(out) # (batch, seq, feature)
        
        # out = out.reshape([batch * seq, 128])
        # out = self.fc(out)
        
        out = out.reshape([batch * seq, 128, 1, 1])
        out = self.fcn(out)
        out = out.reshape([batch, seq, 2])

        return out[:, -1, :]
    
    def save(self, path):
        torch.save({
            'params': self.state_dict(),
            'dropout': self.dropout,
            'num_layers': self.num_layers,
            'name': self.name,
            'seq_length': self.seq_length,
            'width': self.width,
            'height': self.height,
            'loss': self.loss
        }, path)
    
    def load(self, path):
        model_param = torch.load(path)
        self.num_layers = model_param['num_layers']
        self.dropout = model_param['dropout']
        self.seq_length = model_param['seq_length']
        self.loss = model_param['loss']
        
        self.lstm = nn.LSTM(6144, 128, num_layers = self.num_layers, batch_first = True, dropout = self.dropout) 
        
        self.load_state_dict(model_param['params'])
        self.width = model_param['width']
        self.height = model_param['height']
    
    def print_params(self):
        print("Model Type: %s\nImage Size: (%d, %d)\nSequence Length: %d\nNumber Layers: %d\nLoss: %f"
               %(self.name, self.width, self.height, self.seq_length, self.num_layers, self.loss))

class ModelBaseCNN(nn.Module):
    def __init__(self, in_channels, out_channels, in_width = 320, in_height = 240, test_time = False):
        super(ModelBaseCNN, self).__init__()
        # parameters
        self.name = 'cnn'
        self.width = in_width
        self.height = in_height
        self.loss = float('inf')
        
        self.cnn1 = nn.Conv2d(in_channels, 4, [3,4], [1,2], 1) # out = (1, height, width / 2)
      
        rate_x = in_width // 4
        rate_y = in_height // 3
        self.dispcnn = nn.Conv2d(1, 4, 3, 1, 1)

        self.cnn2 = CNN(8, 512, 3, [rate_y, rate_x], 1) # out = (512, 3, 4)
        # self.cnn2 = nn.Conv2d(2, 512, 3, [rate_y, rate_x], 1)
        self.cnn3 = CNN(512, 1024, 3, [3, 4], 1)

        self.cnn4 = nn.Conv2d(1024,out_channels,1, 1, 0)

    def forward(self, rgb, disp):
        batch = rgb.shape[0]
        seq = rgb.shape[1]

        rgb = rgb.reshape([-1, rgb.shape[2], rgb.shape[3], rgb.shape[4]])   # transform (batch, seq, channel, width, height) into (*, channel, width, height)
        disp = disp.reshape([-1, disp.shape[2], disp.shape[3], disp.shape[4]])

        out = self.cnn1(rgb)
        disp = self.dispcnn(disp)
        out = torch.cat((out,disp), 1)

        out = self.cnn2(out)

        out = self.cnn3(out)

        out = self.cnn4(out) # (batch, 1, 1)
        out = out.reshape([-1,2])
        return out
    
    def print_params(self):
        print("Model Type: %s\nImage Size: (%d, %d)\nLoss: %f"
               %(self.name, self.width, self.height, self.loss))
    
    def save(self, path):
        torch.save({
            'params': self.state_dict(),
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'loss': self.loss
        }, path)
    
    def load(self, path):
        model_param = torch.load(path)
        self.load_state_dict(model_param['params'])
        self.width = model_param['width']
        self.height = model_param['height']
        self.loss = model_param['loss']

if __name__ == "__main__":
    model = ModelBaseLSTM(3, 2, 640, 240, num_layers = 2, dropout = 0)
    test_rgb = torch.rand([16,10,3,640,240])
    test_disp = torch.rand([16,10,1,320,240])
    test_out, _ = model(test_rgb, test_disp)
    print(test_out.shape)
    print(test_out)

    time_rgb = torch.rand([1,10,3,640,240])
    time_disp = torch.rand([1,10,1,320,240])

    device = torch.device('cpu')
    time_disp = time_disp.to(device)
    time_rgb = time_rgb.to(device)
    model = model.to(device)

    time_all = np.array([0,0,0,0,0], dtype='float64')
    t1 = time.time()
    count = 100
    for i in range(2000):
        count -= 1
        if count <= 0:
            print("epoch: %d - time pass: %ds" % (i+1, int(time.time() - t1)))
            count = 100
        out, time_array = model(time_rgb, time_disp)
        time_all += time_array
    t2 = time.time() - t1
    print("FPS: %d" % (int(2000/t2)))

    print("reshape: %ds, cnn1: %ds, cnn2: %ds, lstm: %ds, cnn4: %ds" %(
        time_all[0],
        time_all[1],
        time_all[2],
        time_all[3],
        time_all[4]))
