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
# %%

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

class Net_4layers(nn.Module):
    def __init__(self):
        super(Net_4layers, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 11,padding = 5)
        self.conv2 = nn.Conv1d(32,32,3,padding = 1)
        self.conv3 = nn.Conv1d(32,32,3,padding = 1)
        self.conv4 = nn.Conv1d(32,32,3,padding = 1)
        self.conv5 = nn.Conv1d(32,1,3,padding = 1)
        self.pool = nn.MaxPool1d(2,padding = 0)
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        b_s, f_n = x.size()
        x = x.view(b_s, 1, f_n)
        x1 = x[:, :, :512]
        x2 = x[:, :, 512:]
#        print('input layer:')
#        print(x1.size())
        # x2 = x.narrow(2,512,512)
        x1 = self.pool(F.relu(self.conv1(x1)))
#        print('1st layer:')
#        print(x1.size())
        x2 = self.pool(F.relu(self.conv1(x2)))
        x1 = self.pool(F.relu(self.conv2(x1)))
#        print('2nd layer:')
#        print(x1.size())
        x2 = self.pool(F.relu(self.conv2(x2)))
        x1 = self.up(F.relu(self.conv3(x1)))
#        print('3rd layer:')
#        print(x1.size())
        x2 = self.up(F.relu(self.conv3(x2)))
        x1 = self.up(F.relu(self.conv4(x1)))
#        print('4th layer:')
#        print(x1.size())
        x2 = self.up(F.relu(self.conv4(x2)))
        x1 = self.conv5(x1)
#        print('5th layer:')
#        print(x1.size())
        x2 = self.conv5(x2)
        x = torch.cat((x1, x2), dim=2)
        b_s, _, f_n = x.size()
        x = x.view(b_s, -1)
        return x
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
#        print('1st layer:')
#        print(x1.size())
        x2 = self.pool(F.relu(self.conv1(x2)))
        
        x1 = self.pool(F.relu(self.conv2(x1)))
#        print('2nd layer:')
#        print(x1.size())
        x2 = self.pool(F.relu(self.conv2(x2)))
        
        x1 = self.pool(F.relu(self.conv3(x1)))
#        print('3rd layer:')
#        print(x1.size())
        x2 = self.pool(F.relu(self.conv3(x2)))
        
        x1 = self.pool(F.relu(self.conv4(x1)))
#        print('4th layer:')
#        print(x1.size())
        x2 = self.pool(F.relu(self.conv4(x2)))
        
        x1 = self.up(F.relu(self.conv5(x1)))
#        print('5th layer:')
#        print(x1.size())
        x2 = self.up(F.relu(self.conv5(x2)))
        
        x1 = self.up(F.relu(self.conv6(x1)))
#        print('6th layer:')
#        print(x1.size())
        x2 = self.up(F.relu(self.conv6(x2)))
        
        x1 = self.up(F.relu(self.conv7(x1)))
#        print('7th layer:')
#        print(x1.size())
        x2 = self.up(F.relu(self.conv7(x2)))
        
        x1 = self.up(F.relu(self.conv8(x1)))
#        print('8th layer:')
#        print(x1.size())
        x2 = self.up(F.relu(self.conv8(x2)))
        
        x1 = self.conv9(x1)
#        print('9th layer:')
#        print(x1.size())
        x2 = self.conv9(x2)
        
        x = torch.cat((x1, x2), dim=2)
        b_s, _, f_n = x.size()
        x = x.view(b_s, -1)
        
        return x

# %%

def SNR(y_pred):
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
#    max_diff = torch.zeros((batch_size,2,512-1),dtype=torch.double).cuda()
    
    diff = []
    for ii in range(4):
        lag = torch.abs(d[:,:,ii+1:]-d[:,:,:-(ii+1)])
        zero_pad = torch.zeros((batch_size,2,ii)).cuda().double()
        lag_zeros = torch.cat((lag,zero_pad),axis = 2)
        diff.append(lag_zeros.unsqueeze(0))
#        max_diff = torch.max(lag_zeros, max_diff,out=None)
    # 4 * batch_size * 2 *511
    diff = torch.cat(diff, axis=0)    
    
    # 1 * batch * 2 * 1
    mc_thresh = (std_diff*10).unsqueeze(-1).unsqueeze(0)
    # 4 * batch_size * 2 *511
    mask_mc = (diff > mc_thresh).double()
    amp_thresh = (torch.ones(diff.size(),dtype=torch.double)*200).cuda()
    mask_amp = (diff > amp_thresh).double()
    mc_loss = (diff * mask_mc).sum() / (mask_mc.sum()+1e-7)
    amp_loss = (diff * mask_amp).sum() / (mask_amp.sum()+1e-7)
#    max_mc = torch.max(max_diff,mc_thresh,out=None)
#    amp_mc = torch.max(max_diff,amp_thresh,out=None)
#    std_loss_value = torch.max(max_mc+amp_mc)
    return mc_loss + amp_loss


if __name__ == '__main__':
    # %%
    #load data
    np.random.seed(50)
    X = scipy.io.loadmat('Processed_data/noised_HRF_matrix.mat')
    
    X = X['noised_HRF_matrix'];
    Y = scipy.io.loadmat('Processed_data/HRF_profile.mat')
    Y = Y['HRF_profile'];
    X_real_HbO = scipy.io.loadmat('Processed_data/Real_HbO.mat')
    X_real_HbO = X_real_HbO['Real_HbO'];
    X_real_HbR = scipy.io.loadmat('Processed_data/Real_HbR.mat')
    X_real_HbR = X_real_HbR['Real_HbR'];
    X_real = np.concatenate((X_real_HbO,X_real_HbR),axis = 1)
    X = X*1000000
    Y = Y*1000000
    X_real = X_real*1000000
    
    print(X.shape)
    print(X_real.shape)
    # %%
    SampleSize = int(X.shape[0]/2)
    X_HbO = X[0:SampleSize,:]
    X_HbR = X[SampleSize:,:]
    Y_HbO = Y[0:SampleSize,:]
    Y_HbR = Y[SampleSize:,:]
    
    n_train = int(SampleSize*0.8)
    index = np.random.permutation(int(SampleSize))
    X_train_HbO = X_HbO[index[0:n_train],:]
    X_train_HbR = X_HbR[index[0:n_train],:]
    X_train = np.concatenate((X_train_HbO,X_train_HbR),axis=1)
    Y_train_HbO = Y_HbO[index[0:n_train],:]
    Y_train_HbR = Y_HbR[index[0:n_train],:]
    Y_train = np.concatenate((Y_train_HbO,Y_train_HbR),axis=1)
    X_val_HbO = X_HbO[index[n_train:],:]
    X_val_HbR = X_HbR[index[n_train:],:]
    X_val = np.concatenate((X_val_HbO,X_val_HbR),axis=1)
    Y_val_HbO = Y_HbO[index[n_train:],:]
    Y_val_HbR = Y_HbR[index[n_train:],:]
    Y_val = np.concatenate((Y_val_HbO,Y_val_HbR),axis=1)
    #X_train = preprocessing.normalize(X_train)
    
    X_train = X_train[:,:]
    X_val = X_val[:,:]
    Y_train = Y_train[:,:]
    Y_val = Y_val[:,:]
    X_test = X_real[:,:]
    
    train_set = Dataset(X_train, Y_train)
    val_set = Dataset(X_val, Y_val)
    test_set = Dataset(X_test)  

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
            batch_size=32,
            sampler = tordata.SequentialSampler(test_set),
            num_workers=2)
    
    # %% trian and validate
    data_loaders = {"train": trainloader, "val": valloader}
    data_lengths = {"train": n_train, "val": SampleSize-n_train}
    model = ['8layers']
    #loss_modes = ['mse','mse+snr','mse+std','mse+snr+std']
    loss_modes = ['mse+snr+std']
    #torch.set_num_threads(16)
    
    n_epochs = 100
    for model_name in model:
        
        print('Model:', model_name)
        if model_name == '4layers':
            net = Net_4layers().cuda()
        elif model_name == '8layers':
            net = Net_8layers().cuda()
        for loss_mode in loss_modes:
            train_loss1 = []
            val_loss1 = []
            train_loss2 = []
            val_loss2 = []
            train_loss3 = []
            val_loss3 = []
            train_loss = []
            val_loss = []
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            scheduler = StepLR(optimizer, step_size = 100, gamma=0.1)
            lowest_val_loss = 1e6;
            hdf5_filepath = "networks\\" + model_name
            for epoch in range(n_epochs):  # loop over the dataset multiple times
                print('Epoch {}/{}'.format(epoch, n_epochs - 1))
                print('-' * 10)
                for phase in ['train','val']:
                    if phase == 'train':
                        net.train(True)  # Set model to training mode
                    else:
                        net.train(False)
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
                        loss2 = SNR(outputs)
                        loss3 = std_loss(outputs)
                        if loss_mode == 'mse':
                            loss = loss1
                        elif loss_mode == 'mse+snr':
                            loss = loss1 - loss2
                        elif loss_mode == 'mse+std':
                            loss = loss1 + 0.000001*loss3
                        elif loss_mode == 'mse+snr+std':
                            loss = loss1 + 0.1 * (loss2 + 1e5*loss3)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
        
                        # print statistics
                        running_loss1 += loss1.item()
                        running_loss2 += loss2.item()
                        running_loss3 += loss3.item()
                        running_loss += loss.item()
#                        print('mse', loss1)
#                        print('snr', loss2)
#                        print('std', loss3)
#                        print('all', loss)
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
                    print('{} Loss: {:.10f};  STD_all: {:.10f};  std: {:.10f}'.format(
                            phase, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3))
                    scheduler.step()
            
            print('Finished Training')
            plt.figure()
            vl, = plt.plot(val_loss1,'r')
            tl, = plt.plot(train_loss1,'b')
            plt.legend([tl,vl],['training loss', 'validation loss'],)
            figurepath = "figures\\" + model_name+"_"+loss_mode+"_1"+".png"
            plt.savefig(figurepath, transparent=True)
            plt.figure()
            vl, = plt.plot(val_loss2,'r')
            tl, = plt.plot(train_loss2,'b')
            plt.legend([tl,vl],['training loss', 'validation loss'],)
            figurepath = "figures\\" + model_name+"_"+loss_mode+"_2"+".png"
            plt.savefig(figurepath, transparent=True)
            plt.figure()
            vl, = plt.plot(val_loss3,'r')
            tl, = plt.plot(train_loss3,'b')
            plt.legend([tl,vl],['training loss', 'validation loss'],)
            figurepath = "figures\\" + model_name+"_"+loss_mode+"_3"+".png"
            plt.savefig(figurepath, transparent=True)
            plt.figure()
            vl, = plt.plot(val_loss,'r')
            tl, = plt.plot(train_loss,'b')
            plt.legend([tl,vl],['training loss', 'validation loss'],)
            figurepath = "figures\\" + model_name+"_"+loss_mode+"_all"+".png"
            plt.savefig(figurepath, transparent=True)
            print('Finished Fig saving')
            
            trainlosspath = "Processed_data\\train_loss_" + model_name+".txt"
            np.savetxt(trainlosspath, np.array(train_loss), fmt="%s")
            vallosspath = "Processed_data\\val_loss_" + model_name+".txt"
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
            savefilepath = "Processed_data\\Hb_NN_" + model_name+'_'+loss_mode+".mat"
            scipy.io.savemat(savefilepath,{'Y_test': Y_test})
            print('Saved predited data')
