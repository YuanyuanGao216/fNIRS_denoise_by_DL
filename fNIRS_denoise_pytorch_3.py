#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 07 2020
After using the public database

@author: gaoyuanyuan
"""
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import random


random.seed(a = 101, version=2)

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
        self.m = nn.LeakyReLU(negative_slope = 0.00001)

    def forward(self, x):
        b_s, f_n = x.size()
        x = x.view(b_s, 1, f_n)
        x1 = x[:, :, :512]
        x2 = x[:, :, 512:]
        
        x1 = self.pool(self.m(self.conv1(x1)))
        x2 = self.pool(self.m(self.conv1(x2)))
        
        x1 = self.pool(self.m(self.conv2(x1)))
        x2 = self.pool(self.m(self.conv2(x2)))

        x1 = self.pool(self.m(self.conv3(x1)))
        x2 = self.pool(self.m(self.conv3(x2)))

        x1 = self.pool(self.m(self.conv4(x1)))
        x2 = self.pool(self.m(self.conv4(x2)))

        x1 = self.up(self.m(self.conv5(x1)))
        x2 = self.up(self.m(self.conv5(x2)))

        x1 = self.up(self.m(self.conv6(x1)))
        x2 = self.up(self.m(self.conv6(x2)))

        x1 = self.up(self.m(self.conv7(x1)))
        x2 = self.up(self.m(self.conv7(x2)))

        x1 = self.up(self.m(self.conv8(x1)))
        x2 = self.up(self.m(self.conv8(x2)))

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
        zero_pad = torch.zeros((batch_size,2,ii)).double()
        lag_zeros = torch.cat((lag,zero_pad),axis = 2)
        diff.append(lag_zeros.unsqueeze(0))
    # 4 * batch_size * 2 *511
    diff = torch.cat(diff, axis=0)    
    
    # 1 * batch * 2 * 1
    mc_thresh = (std_diff*10).unsqueeze(-1).unsqueeze(0)
    # 4 * batch_size * 2 *511
    mask_mc = (diff > mc_thresh).double()
    amp_thresh = (torch.ones(diff.size(),dtype=torch.double)*200)
    mask_amp = (diff > amp_thresh).double()
    mc_loss = (diff * mask_mc).sum() / (mask_mc.sum()+1e-7)
    amp_loss = (diff * mask_amp).sum() / (mask_amp.sum()+1e-7)

    return mc_loss + amp_loss
# %% load data

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

RealData = scipy.io.loadmat('Processed_data/RealData.mat')
dc = RealData['dc']

X_real_HbO = np.transpose(dc[:,0,:])
X_real_HbO = X_real_HbO[:,0: 512*(X_real_HbO.shape[1] // 512)]
X_real_HbO = np.reshape(X_real_HbO,(-1,512))

X_real_HbR = np.transpose(dc[:,1,:])
X_real_HbR = X_real_HbR[:,0: 512*(X_real_HbR.shape[1] // 512)]
X_real_HbR = np.reshape(X_real_HbR,(-1,512))

X_real = np.concatenate((X_real_HbO,X_real_HbR),axis = 1)

X_train = X_train*1000000
X_val = X_val*1000000
X_test = X_test*1000000

Y_train = Y_train*1000000
Y_val = Y_val*1000000
Y_test = Y_test*1000000

X_real = X_real*1000000


X_train = X_train[:,:]
Y_train = Y_train[:,:]
X_val = X_val[:,:]
Y_val = Y_val[:,:]
X_test = X_test[:,:]
Y_test = Y_test[:,:]
X_real = X_real[:,:]

train_set = Dataset(X_train, Y_train)
val_set = Dataset(X_val, Y_val)
test_set = Dataset(X_test, Y_test)  
real_set = Dataset(X_real)
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
# %% trian and validate
data_loaders = {"train": trainloader, "val": valloader}


n_epochs = 1

net = Net_8layers()
train_loss1 = []
val_loss1 = []
train_loss2 = []
val_loss2 = []
train_loss3 = []
val_loss3 = []
train_loss = []
val_loss = []
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size = 20, gamma=0.1)
lowest_val_loss = 1e6;
hdf5_filepath = "networks/8layers"
# %%
for epoch in range(n_epochs):
    for phase in ['train','val']:
        if phase == 'train':
            net.train()
        else:
            net.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss  = 0.0
        for i, data in enumerate(data_loaders[phase], 0):
            inputs, y_true = data
            optimizer.zero_grad()
            outputs = net(inputs.float())
            outputs = outputs.double()
            mse_loss = nn.MSELoss()
            loss1 = mse_loss(outputs, y_true)
            loss2 = SNR_loss(outputs)
            loss3 = std_loss(outputs)

            loss = loss1 + 0.5 * loss2 + 5 * loss3

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
        # print('{} Loss: {:.5f};  loss1: {:.5f};  loss2: {:.5f}；loss3: {:.5f}'.format(
        #         phase, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3))
    scheduler.step()


plt.figure()
vl, = plt.plot(val_loss1,'r')
tl, = plt.plot(train_loss1,'b')
plt.legend([tl,vl],['training loss', 'validation loss'],)
plt.title('loss #1')
figurepath = "/Figures/loss1.png"
plt.savefig(figurepath, transparent=True)

plt.figure()
vl, = plt.plot(val_loss2,'r')
tl, = plt.plot(train_loss2,'b')
plt.legend([tl,vl],['training loss', 'validation loss'],)
plt.title('loss #2')
figurepath = "/Figures/loss2.png"
plt.savefig(figurepath, transparent=True)

plt.figure()
vl, = plt.plot(val_loss3,'r')
tl, = plt.plot(train_loss3,'b')
plt.legend([tl,vl],['training loss', 'validation loss'],)
plt.title('loss #3')
figurepath = "/Figures/loss3.png"
plt.savefig(figurepath, transparent=True)

plt.figure()
vl, = plt.plot(val_loss,'r')
tl, = plt.plot(train_loss,'b')
plt.legend([tl,vl],['training loss', 'validation loss'],)
plt.title('loss total')
figurepath = "/Figures/total_loss.png"
plt.savefig(figurepath, transparent=True)

trainlosspath = "/Processed_data/train_loss.txt"
np.savetxt(trainlosspath, np.array(train_loss), fmt="%s")
vallosspath = "/Processed_data/val_loss.txt"
np.savetxt(vallosspath, np.array(val_loss), fmt="%s")

net.load_state_dict(torch.load(hdf5_filepath))

Y_test_predict = []
for i, data in enumerate(testloader, 0):
    inputs = data[0]
    outputs = net(inputs.float())
    Y_test_predict.append(outputs)
    

Y_test_predict = np.concatenate(Y_test_predict, axis=0)
Y_test_predict = Y_test_predict/1000000
savefilepath = "Processed/Test_NN.mat"
scipy.io.savemat(savefilepath,{'Y_test_predict': Y_test_predict})

plt.figure()
X_test_example = X_test[0,:]/1000000
Y_test_example = Y_test[0,:]/1000000
Y_test_predict_example = Y_test_predict[0,:]
X_test_line, = plt.plot(X_test_example,'b')
Y_test_line, = plt.plot(Y_test_example,'r')
Y_test_predict_line, = plt.plot(Y_test_predict_example,'g')
plt.title('test_example')
plt.legend([X_test_line,Y_test_line,Y_test_predict_line],['X', 'Y', 'Y_predict'],)


Y_real_predict = []
for i, data in enumerate(realloader, 0):
    inputs = data[0]
    outputs = net(inputs.float())
    Y_real_predict.append(outputs)
Y_real_predict = np.concatenate(Y_real_predict, axis=0)
Y_real_predict = Y_real_predict/1000000
savefilepath = "Processed/Real_NN.mat"
scipy.io.savemat(savefilepath,{'Y_real_predict': Y_real_predict})

 print('Y_real shape is ',Y_real.shape)

plt.figure()
X_real_example = X_real[0,:]/1000000
Y_real_example = Y_real[0,:]
X_real_line, = plt.plot(X_real_example,'b')
Y_real_line, = plt.plot(Y_real_example,'r')
plt.title('real_example')
plt.legend([X_real_line,Y_real_line],['X', 'Y'],)
figurepath = "/content/gdrive/My Drive/Y_real_example.png"
plt.savefig(figurepath, transparent=True)