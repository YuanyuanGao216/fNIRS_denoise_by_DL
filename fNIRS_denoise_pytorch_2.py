#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:35:51 2020

@author: gaoyuanyuan
"""
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tordata
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import time

class Dataset(tordata.Dataset):
    def __init__(self, noisy_data, clean_data=None):
        self.noisy_data = noisy_data
        self.clean_data = clean_data
    
    def __getitem__(self, index):
        noisy_data = self.noisy_data[index]
        clean_data = -1
        if self.clean_data is not None:
            clean_data = self.clean_data[index]
        return noisy_data, clean_data
    
    def __len__(self):
        return self.noisy_data.shape[0]

# %% define nn models

class Net_8layers(nn.Module):
    def __init__(self):
        super(Net_8layers, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 11,padding = 5)
        self.conv2 = nn.Conv1d(32,32,3,padding = 1)
        self.conv3 = nn.Conv1d(32,32,3,padding = 1)
        self.conv4 = nn.Conv1d(32,32,3,padding = 1)
        self.conv5 = nn.Conv1d(32,32,3,padding = 1)
        self.conv6 = nn.Conv1d(32,32,3,padding = 1)
        self.conv7 = nn.Conv1d(32,32,3,padding = 1)
        self.conv8 = nn.Conv1d(32,32,3,padding = 1)
        self.conv9 = nn.Conv1d(32,1,3,padding = 1)
        self.pool = nn.MaxPool1d(2,padding = 0)
        self.up = nn.Upsample(scale_factor=2)
        self

    def forward(self, x):
        b_s, f_n = x.size()
        x = x.view(b_s, 1, f_n)
        x1 = x[:, :, :512]
        x2 = x[:, :, 512:]
        
        x1 = self.pool(F.relu(self.conv1(x1)))
        x2 = self.pool(F.relu(self.conv1(x2)))
        
        x1 = self.pool(F.relu(self.conv2(x1)))
        x2 = self.pool(F.relu(self.conv2(x2)))
        
        x1 = self.pool(F.relu(self.conv3(x1)))
        x2 = self.pool(F.relu(self.conv3(x2)))
        
        x1 = self.pool(F.relu(self.conv4(x1)))
        x2 = self.pool(F.relu(self.conv4(x2)))
        
        x1 = self.up(F.relu(self.conv5(x1)))
        x2 = self.up(F.relu(self.conv5(x2)))
        
        x1 = self.up(F.relu(self.conv6(x1)))
        x2 = self.up(F.relu(self.conv6(x2)))
        
        x1 = self.up(F.relu(self.conv7(x1)))
        x2 = self.up(F.relu(self.conv7(x2)))
        
        x1 = self.up(F.relu(self.conv8(x1)))
        x2 = self.up(F.relu(self.conv8(x2)))
        
        x1 = self.conv9(x1)
        x2 = self.conv9(x2)
        
        x = torch.cat((x1, x2), dim=2)
        b_s, _, f_n = x.size()
        x = x.view(b_s, -1)
        
        return x
    
def SNR_loss(y_pred):
    batch_size, f_n = y_pred.size()
    y_pred = y_pred.view(batch_size, 2, -1)
    SNR = torch.std(y_pred, axis=2).mean()
    return SNR

def std_loss(y_pred):
    batch_size, f_n = y_pred.size()
    y_pred = y_pred/1000000
    y_pred = y_pred.view(batch_size, 1, f_n)
    HbO = y_pred[:, :, :512]
    HbR = y_pred[:, :, 512:]
    d1 = 148*HbO+384*HbR
    d2 = 252*HbO+179*HbR
    d = torch.cat((d1,d2),axis = 1)
    d = d*2.376*6
    # batch_size * 2
    std_diff = torch.std(d[:,:,1:]-d[:,:,:-1],axis = 2)
    
    diff = []
    for ii in range(4):
        lag = torch.abs(d[:,:,ii+1:]-d[:,:,:-(ii+1)])
        zero_pad = torch.zeros((batch_size,2,ii)).cuda().double()
        lag_zeros = torch.cat((lag,zero_pad),axis = 2)
        diff.append(lag_zeros.unsqueeze(0))
    # 4 * batch_size * 2 *511
    diff = torch.cat(diff, axis=0)    
    
    # 1 * batch * 2 * 1
    mc_thresh = (std_diff*20).unsqueeze(-1).unsqueeze(0)
    # 4 * batch_size * 2 *511
    mask_mc = (diff > mc_thresh).double()
    amp_thresh = (torch.ones(diff.size(),dtype=torch.double)*0.3).cuda()
    mask_amp = (diff > amp_thresh).double()
    mc_loss = (diff * mask_mc).sum() / (mask_mc.sum()+1e-7)
    amp_loss = (diff * mask_amp).sum() / (mask_amp.sum()+1e-7)

    return mc_loss + amp_loss
# %% load data
if __name__ == '__main__':
    np.random.seed(50)
    
    SimulateData = scipy.io.loadmat('Processed_data/SimulateData.mat')
    
    X_train = SimulateData['HRF_train_noised']
    X_train = X_train.reshape((-1, 512))
    n = X_train.shape[0]
    X_train = np.concatenate((X_train[0:int(n/2),:],X_train[int(n/2):,:]),axis = 1)
    
    X_val = SimulateData['HRF_val_noised']
    X_val = X_val.reshape((-1, 512))
    n = X_val.shape[0];
    X_val = np.concatenate((X_val[0:int(n/2),:],X_val[int(n/2):,:]),axis = 1)
    
    X_test = SimulateData['HRF_test_noised']
    X_test = X_test.reshape((-1, 512))
    n = X_test.shape[0];
    X_test = np.concatenate((X_test[0:int(n/2),:],X_test[int(n/2):,:]),axis = 1)
    
    Y_train = SimulateData['HRF_train']
    Y_train = Y_train.reshape((-1, 512))
    n = Y_train.shape[0]
    Y_train = np.concatenate((Y_train[0:int(n/2),:],Y_train[int(n/2):,:]),axis = 1)
    
    Y_val = SimulateData['HRF_val']
    Y_val = Y_val.reshape((-1, 512))
    n = Y_val.shape[0];
    Y_val = np.concatenate((Y_val[0:int(n/2),:],Y_val[int(n/2):,:]),axis = 1)
    
    Y_test = SimulateData['HRF_test']
    Y_test = Y_test.reshape((-1, 512))
    n = Y_test.shape[0];
    Y_test = np.concatenate((Y_test[0:int(n/2),:],Y_test[int(n/2):,:]),axis = 1)
    
    # %% here need to change; the real finrs dc data need to be saved, the leave out one
    RealData = scipy.io.loadmat('Processed_data/RealData.mat')
    net_input = RealData['net_input']
    dc = net_input['dc'] # (1,8)
    dc_act = net_input['dc_act'] # (1,8)
    X_real = []
    X_real_act = []
    
    number_array = []
    for i in range(8):
        Hb = dc[0,i]
        
        n = Hb.shape[0] // 512
        
        HbO = np.transpose(np.squeeze(Hb[:n*512, 0, :]))
        HbR = np.transpose(np.squeeze(Hb[:n*512, 1, :]))
        
        HbO = np.transpose(np.reshape(HbO,(512,-1)))
        HbR = np.transpose(np.reshape(HbR,(512,-1)))
        
        X = np.concatenate((HbO,HbR),axis = 1)
        number_array.append(X.shape[0])
        X_real.append(X)
        
        Hb_act = dc_act[0,i]
        n = Hb.shape[0] // 512
        
        HbO_act = np.transpose(np.squeeze(Hb_act[:n*512, 0, :]))
        HbR_act = np.transpose(np.squeeze(Hb_act[:n*512, 1, :]))
        
        HbO_act = np.transpose(np.reshape(HbO_act,(512,-1)))
        HbR_act = np.transpose(np.reshape(HbR_act,(512,-1)))
        
        X = np.concatenate((HbO_act,HbR_act),axis = 1)
        X_real_act.append(X)
    
    X_real = np.concatenate(X_real, axis=0)
    X_real_act = np.concatenate(X_real_act, axis=0)
    savefilepath = "Processed_data/number_array.mat"
    scipy.io.savemat(savefilepath,{'number_array': number_array})
    # %% double check the magnitude 
    X_train = X_train*1000000
    X_val = X_val*1000000
    X_test = X_test*1000000
    
    Y_train = Y_train*1000000
    Y_val = Y_val*1000000
    Y_test = Y_test*1000000
    
    X_real = X_real*1000000
    X_real_act = X_real_act*1000000
    
    X_train = X_train[:,:]
    Y_train = Y_train[:,:]
    X_val = X_val[:,:]
    Y_val = Y_val[:,:]
    X_test = X_test[:,:]
    Y_test = Y_test[:,:]
    X_real = X_real[:,:]
    X_real_act = X_real_act[:,:]
    
    train_set = Dataset(X_train, Y_train)
    val_set = Dataset(X_val, Y_val)
    test_set = Dataset(X_test, Y_test)  
    real_set = Dataset(X_real)
    real_set_act = Dataset(X_real_act)
    
    # %% define data loaders
    trainloader = torch.utils.data.DataLoader(
            dataset = train_set,
            batch_size=512,
            sampler = tordata.RandomSampler(train_set),
            num_workers=2)
    
    valloader = torch.utils.data.DataLoader(
            dataset = val_set, 
            batch_size=512,
            sampler = tordata.SequentialSampler(val_set),
            num_workers=2)
    
    testloader = torch.utils.data.DataLoader(
            dataset = test_set, 
            batch_size=512,
            sampler = tordata.SequentialSampler(test_set),
            num_workers=2)
    
    realloader = torch.utils.data.DataLoader(
            dataset = real_set, 
            batch_size=32,
            sampler = tordata.SequentialSampler(real_set),
            num_workers=2)
    
    realloader_act = torch.utils.data.DataLoader(
            dataset = real_set_act, 
            batch_size=32,
            sampler = tordata.SequentialSampler(real_set_act),
            num_workers=2)
    # %% trian and validate
    data_loaders = {"train": trainloader, "val": valloader}
    # model = ['8layers']
    model_name = '8layers'
    n_epochs = 100
    print('start')
    
    net = Net_8layers().cuda()
    train_loss1 = []
    val_loss1 = []
    train_loss2 = []
    val_loss2 = []
    train_loss3 = []
    val_loss3 = []
    train_loss = []
    val_loss = []
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size = 25, gamma=0.1)
    lowest_val_loss = 1e6;
    hdf5_filepath = "networks/8layers"
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)
        for phase in ['train','val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_loss = 0.0
            for i, data in enumerate(data_loaders[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, y_true = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs.cuda().float())
                outputs = outputs.double()
                mse_loss = nn.MSELoss()
                loss1 = mse_loss(outputs, y_true.cuda())
                loss2 = SNR_loss(outputs)
                loss3 = std_loss(outputs)
                
#                 loss = loss1 + 0.01 *(loss2 + loss3)
                loss = loss1 + 1 * loss2 + 100*loss3

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
                running_loss += loss.item()
            epoch_loss1 = running_loss1 / len(data_loaders[phase])
            epoch_loss2 = running_loss2 / len(data_loaders[phase])
            epoch_loss3 = running_loss3 / len(data_loaders[phase])
            epoch_loss = running_loss / len(data_loaders[phase])
            if phase == 'train':
                train_loss1.append(epoch_loss1)
                train_loss2.append(epoch_loss2)
                train_loss3.append(epoch_loss3)
                train_loss.append(epoch_loss)  # Set model to training mode
            else:
                val_loss1.append(epoch_loss1)
                val_loss2.append(epoch_loss2)
                val_loss3.append(epoch_loss3)
                val_loss.append(epoch_loss)
                if epoch_loss < lowest_val_loss:
                    lowest_val_loss = epoch_loss
                    torch.save(net.state_dict(), hdf5_filepath)
            print('{} Loss: {:.5f};  loss1: {:.5f};  loss2: {:.5f}；loss3: {:.5f}'.format(
                    phase, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3))
        scheduler.step()

    print('Finished Training')
    plt.figure()
    vl, = plt.plot(val_loss1,'r')
    tl, = plt.plot(train_loss1,'b')
    plt.legend([tl,vl],['training loss', 'validation loss'],)
    plt.title('loss1')
    figurepath = "Figures/" + model_name+"_1"+".png"
    plt.savefig(figurepath, transparent=True)
    
    plt.figure()
    vl, = plt.plot(val_loss2,'r')
    tl, = plt.plot(train_loss2,'b')
    plt.legend([tl,vl],['training loss', 'validation loss'],)
    plt.title('loss2')
    figurepath = "Figures/" + model_name+"_2"+".png"
    plt.savefig(figurepath, transparent=True)
    
    plt.figure()
    vl, = plt.plot(val_loss3,'r')
    tl, = plt.plot(train_loss3,'b')
    plt.legend([tl,vl],['training loss', 'validation loss'],)
    plt.title('loss3')
    figurepath = "Figures/" + model_name+"_3"+".png"
    plt.savefig(figurepath, transparent=True)
    
    plt.figure()
    vl, = plt.plot(val_loss,'r')
    tl, = plt.plot(train_loss,'b')
    plt.legend([tl,vl],['training loss', 'validation loss'],)
    plt.title('total loss')
    figurepath = "Figures/" + model_name+"_all"+".png"
    plt.savefig(figurepath, transparent=True)
    print('Finished Fig saving')

    trainlosspath = "Processed_data/train_loss_" + model_name+".txt"
    np.savetxt(trainlosspath, np.array(train_loss), fmt="%s")
    vallosspath = "Processed_data/val_loss_" + model_name+".txt"
    np.savetxt(vallosspath, np.array(val_loss), fmt="%s")
    print('Finished writing loss files')

    net.load_state_dict(torch.load(hdf5_filepath))
    print('loaded nn file')
    
    
    Y_test = []
    for i, data in enumerate(testloader, 0):
        inputs = data[0]
        outputs = net(inputs.cuda().float())
        Y_test.append(outputs.cpu().data.numpy())
        
    
    Y_test = np.concatenate(Y_test, axis=0)
    Y_test = Y_test/1000000
    savefilepath = "Processed_data/Test_NN_" + model_name+".mat"
    scipy.io.savemat(savefilepath,{'Y_test': Y_test})
    
    print('Y_test shape is ',Y_test.shape)
    
    plt.figure()
    X_test_example = X_test[0,:];
    Y_test_example = Y_test[0,:]*1000000;
    X, = plt.plot(X_test_example,'b')
    Y, = plt.plot(Y_test_example,'r')
    figurepath = "Figures/Y_test_example.png"
    plt.savefig(figurepath, transparent=True)
    
    Y_real = []
    for i, data in enumerate(realloader, 0):
        inputs = data[0]
        outputs = net(inputs.cuda().float())
        Y_real.append(outputs.cpu().data.numpy())
    Y_real = np.concatenate(Y_real, axis=0)
    Y_real = Y_real/1000000
    savefilepath = "Processed_data/Real_NN_" + model_name+".mat"
    scipy.io.savemat(savefilepath,{'Y_real': Y_real})
    
    print('Y_real shape is ',Y_real.shape)
    
    plt.figure()
    X_real_example = X_real[0,:]
    Y_real_example = Y_real[0,:]*1000000
    X, = plt.plot(X_real_example,'b')
    Y, = plt.plot(Y_real_example,'r')
    figurepath = "Figures/Y_real_example.png"
    plt.savefig(figurepath, transparent=True)
    
    Y_real_act = []
    for i, data in enumerate(realloader_act, 0):
        inputs = data[0]
        outputs = net(inputs.cuda().float())
        Y_real_act.append(outputs.cpu().data.numpy())
    Y_real_act = np.concatenate(Y_real_act, axis=0)
    Y_real_act = Y_real_act/1000000
    savefilepath = "Processed_data/Real_NN_" + model_name+"_act.mat"
    scipy.io.savemat(savefilepath,{'Y_real_act': Y_real_act})
    
    plt.figure()
    X_real_example = X_real_act[0,:]
    Y_real_example = Y_real_act[0,:]*1000000
    X, = plt.plot(X_real_example,'b')
    Y, = plt.plot(Y_real_example,'r')
    figurepath = "Figures/Y_real_example_act.png"
    plt.savefig(figurepath, transparent=True)
    print('Y_real_act shape is ',Y_real_act.shape)
    
    
    
    print('Saved predited data')