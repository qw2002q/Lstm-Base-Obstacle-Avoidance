###
# Author: Seya Peng
###
import torch
import torchvision
import numpy as np
import torch.nn as nn
import time
import datetime
import math
import matplotlib.pyplot as plt

# show loss plot
def loss_plot_show(loss_plot, label = "loss", color='blue'):
    x = range(len(loss_plot))

    plt.plot(x, loss_plot, label = label, color = color)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    
def show_histogram(data, label_list, title, color='blue'):
    x = range(len(data))
    rects = plt.bar(height=data, width=0.45, alpha=0.8, color=color, label=title, x=x)
    
    plt.xlabel("loss range")
    plt.ylabel("frequency")
    plt.xticks(x, label_list)
    plt.title(title)
    plt.legend()
    plt.show()

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.pause(5)
    

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            # torch.nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.LSTM):
            for i in range(len(m.all_weights)):
                nn.init.xavier_normal(m.all_weights[i][0], gain = 1)
                nn.init.xavier_normal(m.all_weights[i][1], gain = 1)

## Discarded ##
def weights_init_normal(m):
    '''
        initial the weight in normal distribution
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
def timeSince(since):
    now = time.time()
    s = now - since
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)

def timeShow(s):
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%dh %dm %ds' % (h, m, s)
    
    
def timeSince_ms(since):
    now = datetime.datetime.now()
    out = now - since
    return '%d.%ds' % (out.seconds, out.microseconds)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# to load the parameters
class ArgvList:
    def __init__(self, argv):
        self.optional = ['model_type', 'save_path', 'batch_size', 'n_epochs',
                    'seq_length', 'num_layers', 'device', 'lr', 'dataloader_path', 'load_path', 
                    'view', 'start', 'end', 'pretrain_model']
        self.argvList = {}
        for i in range(1, len(argv)):
            param = argv[i].split('=')
            if(not len(param) == 2):
                self.printError()
                exit()
            if(param[0].lower() not in self.optional):
                self.printError()
                exit()
            self.argvList[param[0].lower()] = param[1]
        
    def getargv(self, name, default = 0):
        if(name.lower() in self.argvList):
            return self.argvList[name]
        return default
    
    def printError(self):
        print("Param input Error!!!")
        print("Example: python train_model.py model_name=lstm batch_size=16")
        print("[optional parameters]\n"+
                "model_type(lstm or cnn)\n"+
                "save_path\n"+
                "load_path\n"+
                "batch_size\n"+
                "n_epochs\n"+
                "seq_length\n"+
                "num_layers\n"+
                "device(cuda:0 or cpu)\n"+
                "lr\n"+
                "dataloader_path\n"+
                "view(True or False)[to view the test result]\n"+
                "start\n"+
                "end\n"+
                "pretrain_model")
        
        
    