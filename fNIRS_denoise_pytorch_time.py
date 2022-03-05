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
import torch.utils.data as tordata
import time


class Dataset(tordata.Dataset):
    def __init__(self, noisy_data, clean_data = None):
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
def main():
    np.random.seed(50)
    X_test = scipy.io.loadmat('Processed_data/HRF_test_noised.mat')
    X_test = X_test['HRF_test_noised'];
    n = X_test.shape[0];
    X_test = np.concatenate((X_test[0:int(n/2),:],X_test[int(n/2):,:]),axis = 1)
    Y_test = scipy.io.loadmat('Processed_data/HRF_test.mat')
    Y_test = Y_test['HRF_test'];
    n = Y_test.shape[0];
    Y_test = np.concatenate((Y_test[0:int(n/2),:],Y_test[int(n/2):,:]),axis = 1)
    X_test = X_test*1000000
    Y_test = Y_test*1000000
    X_test = X_test[0:100,:]
    Y_test = Y_test[0:100,:]
    test_set = Dataset(X_test, Y_test)  

    testloader = torch.utils.data.DataLoader(
            dataset = test_set, 
            batch_size = 512,
            sampler = tordata.SequentialSampler(test_set),
            num_workers = 2)

    net = Net_8layers()
    hdf5_filepath = "networks/8layers"
    net.load_state_dict(torch.load(hdf5_filepath, map_location=torch.device('cpu')))
    print('loaded nn file')
    for i, data in enumerate(testloader, 0):
        print(i)
        inputs = data[0]
        inputs = inputs.float()
        start_time = time.time()
        net(inputs)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    return
# %%
if __name__ == '__main__':
    main()