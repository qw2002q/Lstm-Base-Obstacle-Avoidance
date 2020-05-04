###
#  Author: Seya Peng
#  Train Model Function
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

def train(dataloader, model, criterion, optimizer, n_epochs, lr = 0.05, device = torch.device('cpu'), init_weight = True, save_flag = True, slow_down = False, slow_step=10, slow_rate=0.5, save_name='trained_model.pth'):
    if init_weight: ## unfinished ##
        model.apply(initialize_weights)
        print("Init Weight Compele")

    model = model.to(device)
    criterion = criterion.to(device)
    # optimizer = optimizer
    model.train()

    t2 = time.time()
    loss_plot = []
    min_loss = model.loss
    
    # adjust_learning_rate(optimizer, lr)    # set learning_rate
    if slow_down:
         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=slow_step,gamma=slow_rate)

    # temp for adjust lr 1
    # lr_list = [0.05, 0.01, 0.005, 0.001]
    # epoch_list = [20, 40, 6099999]
    # point = 0
    # adjust_learning_rate(optimizer, lr_list[0])

    for epoch in range(n_epochs):
        # temp for adjust lr 2
        # if(epoch > epoch_list[point]):
        #     adjust_learning_rate(optimizer, lr_list[(point + 1)])
        #     point += 1
        
        loss_all = 0
        for batch in dataloader:
            t = datetime.datetime.now()

            img = batch['image'].float().to(device)
            gt = batch['gt'].float().to(device)
            disp = batch['disp'].float().to(device)

            optimizer.zero_grad()

            output = model(img, disp)
            # output *= torch.tensor([1,10]).float().to(device)
            # gt *= torch.tensor([1,10]).float().to(device)
            # loss = criterion(output, gt)
            # loss1=0
            # loss2=0
            loss1 = criterion(output[:, 0], gt[:, 0])
            loss2 = criterion(output[:, 1], gt[:, 1])
            loss = loss1 + loss2 * 10
            
            # print(gt)
            # print(output)
            # print(output[:, 1])
            # print(gt[:, 1])
            # print('loss: %f\n loss1: %.4f, loss2: %.4f' 
            #     %(
            #         loss,
            #         loss1,
            #         loss2
            #     )
            # )
            # print("====================")

            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        # Print log
        epoch_loss = loss_all / len(dataloader)
        loss_plot.append(epoch_loss)
        if epoch % 1 == 0:
            print("Time:" + timeSince(t2) + 
            " || Left Time:" + timeShow((time.time() - t2) / (epoch + 1) * (n_epochs - epoch - 1)))
            print(
                "\r[Epoch %d/%d] [Batch %d] [train_loss %f]"
                % (
                       epoch + 1,
                       n_epochs,
                       len(dataloader),
                       epoch_loss
                )
            )
        if slow_down:
            scheduler.step()
        if save_flag:
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                model.loss = min_loss
                save_path = os.path.dirname(os.path.abspath(__file__))+"/../../models/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                model.save(save_path + save_name)
                # torch.save(model,save_path + "trained_model.pth")
                # torch.save(model.state_dict(),save_path + "trained_model.pth")
                print("save model with loss: %f" % (epoch_loss))
    return loss_plot 