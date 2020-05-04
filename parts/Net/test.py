###
#  Author: Seya Peng
#  Test Model Function
###

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import datetime

## import Superior documents
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from ulits import *
sys.path.pop()  # resume after using

def test(dataloader, model, criterion, device = torch.device('cpu'), sviewr = None, start = 0, end = -1):
    print("==== Test model ====")
    model = model.to(device)
    criterion = criterion.to(device)
    model.eval()
    
    if(end <= 0): 
        end = int(len(dataloader))
    print("(start = %d, end = %d)" % (start, end))
    
    with torch.no_grad():
        for epoch in range(1):
            loss_all = 0
            loss1_list = []
            loss2_list = []
            tongji1_list = [0,0,0,0,0,0]
            tongji2_list = [0,0,0,0,0,0,0,0]
            loss_list = []
            t2 = time.time()
            count = 0;
            for (index, batch) in enumerate(dataloader):
                if(index < start):
                    continue
                if(index >= end):
                    break
                    
                img = batch['image'].float().to(device)
                gt = batch['gt'].float().to(device)
                disp = batch['disp'].float().to(device)
                path_string = batch['path_string'];

                output = model(img, disp)
                # output *= torch.tensor([1,10]).float().to(device)
                # gt *= torch.tensor([1,10]).float().to(device)

                loss1 = criterion(output[:, 0], gt[:, 0])
                loss2 = criterion(output[:, 1], gt[:, 1])
                loss = loss1 + loss2 * 10
                
                loss_all += loss.item()
                
                out1, out2 = getSubMean(output, gt)
                if(out1 > 0.45): out1 = 0.44
                loss1_list.append(out1)
                loss2_list.append(out2)
                tongji1_list[tongJi(out1, [0.05,0.10,0.15,0.20,0.25])] += 1
                tongji2_list[tongJi(out2,[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])] += 1
                loss_list.append(loss.item())
                count += 1

                # output *= torch.tensor([1,0.1]).float().to(device)
                # gt *= torch.tensor([1,0.1]).float().to(device)
                
                if(sviewr):
                    out = output.cpu().detach().numpy()
                    r = gt.cpu().detach().numpy()
                    sviewr.run(path_string[0], -out[0][1], out[0][0], -r[0][1], r[0][0])
                
                print("%d/%d" % (index + 1 - start, end - start))
                print(gt)
                print(output)
                print('loss: %f\nloss1: %.4f, loss2: %.4f' 
                    %(
                        loss,
                        loss1,
                        loss2
                    )
                )
                print('===============================')
                # sviewr.run(path_string[0], output[0][1], output[0][0], gt[0][1], gt[0][0])
                
                if(count >= 40):
                    print("FPS: %d" 
                        % (
                            count / (time.time() - t2),
                        )
                    )
                    count = 0
                    t2 = time.time()
            print('[test_loss: %f]' 
                    %(
                        loss_all / (end - start)
                    )
                )
            return loss_all / (end - start), loss1_list, loss2_list, loss_list, tongji1_list, tongji2_list

def getSubMean(a, b):
    len = a.shape[0]
    out1 = 0
    out2 = 0
    for i in range(len):
        out1 += abs(a[i][0] - b[i][0])
        out2 += abs(a[i][1] - b[i][1])
    out1 /= len
    out2 /= len
    return out1, out2

def tongJi(num, tongji):
    for i in range(len(tongji)):
        if(num < tongji[i]):
            return i
    else:
        return len(tongji)
    