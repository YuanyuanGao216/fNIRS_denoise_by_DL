#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:53:55 2020

@author: gaoyuanyuan
"""

from keras.models import Model
from keras.layers import Conv1D, Input,MaxPooling1D,UpSampling1D,Dropout
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
from keras import optimizers
import scipy.io
from sklearn import preprocessing
# %% data genetator
def TrainGenerator(train_x,train_y):
    x_size = len(train_x)
    x_idx = np.asarray([i for i in range(x_size)])
    
    batch_size = 32
    batch_start = 0
    new_batch = False
    while 1:
        
        if (batch_start + 1) * batch_size > x_size:
            batch_start = 0
            new_batch = True
            
        if new_batch:
            x_idx = np.random.permutation(x_idx)
            new_batch = False
            
        indexes = x_idx[batch_start * batch_size : (batch_start + 1) * batch_size] 
        indexes = np.array(indexes,dtype=int)
        x = train_x[indexes, :,:]
        y = train_y[indexes, :,:]
        x = np.array(x)
        y = np.array(y)
        batch_start += 1
        yield (x, y)
def ValGenerator(train_x,train_y):
    x_size = len(train_x)
    x_idx = np.asarray([i for i in range(x_size)])
    
    batch_size = 8
    batch_start = 0
    new_batch = False
    while 1:
        
        if (batch_start + 1) * batch_size > x_size:
            batch_start = 0
            new_batch = True
            
        if new_batch:
            x_idx = np.random.permutation(x_idx)
            new_batch = False
            
        indexes = x_idx[batch_start * batch_size : (batch_start + 1) * batch_size] 
        indexes = np.array(indexes,dtype=int)
        x = train_x[indexes, :,:]
        y = train_y[indexes, :,:]
        x = np.array(x)
        y = np.array(y)
        batch_start += 1
        yield (x, y)
# %% model arch

def L4_arch():
    fNIRS_input = Input(shape = (512,1))
    
    c1 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c1')(fNIRS_input)
    p1 = MaxPooling1D(pool_size=2, padding='same',name = 'p1')(c1)
    
    c2 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c2')(p1)
    p2 = MaxPooling1D(pool_size=2,padding='same',name = 'p2')(c2)
    
    c3 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c3')(p2)
    u1 = UpSampling1D(size=2,name = 'u1')(c3)
    
    c4 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c4')(u1)
    u2 = UpSampling1D(size=2, name = 'u2')(c4)
    
    HRF_output = Conv1D(1,3,padding = 'same',activation = 'linear',name = 'output')(u2)
    
    model = Model(fNIRS_input,HRF_output)
    
    return model
def L4_dropout_arch():
    fNIRS_input = Input(shape = (512,1))
    
    c1 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c1')(fNIRS_input)
    p1 = MaxPooling1D(pool_size=2, padding='same',name = 'p1')(c1)
    d1 = Dropout(0.1)(p1)
    
    c2 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c2')(d1)
    p2 = MaxPooling1D(pool_size=2,padding='same',name = 'p2')(c2)
    d2 = Dropout(0.1)(p2)
    
    c3 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c3')(d2)
    u1 = UpSampling1D(size=2,name = 'u1')(c3)
    d3 = Dropout(0.1)(u1)
    
    c4 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c4')(d3)
    u2 = UpSampling1D(size=2, name = 'u2')(c4)
    d4 = Dropout(0.1)(u2)
    
    HRF_output = Conv1D(1,3,padding = 'same',activation = 'linear',name = 'output')(d4)
    
    model = Model(fNIRS_input,HRF_output)
    
    return model
def L8_arch():
    fNIRS_input = Input(shape = (512,1))
    
    c1 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c1')(fNIRS_input)
    p1 = MaxPooling1D(pool_size=2, padding='same',name = 'p1')(c1)
    
    c2 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c2')(p1)
    p2 = MaxPooling1D(pool_size=2,padding='same',name = 'p2')(c2)
    
    c3 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c3')(p2)
    p3 = MaxPooling1D(pool_size=2, padding='same',name = 'p3')(c3)
    
    c4 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c4')(p3)
    p4 = MaxPooling1D(pool_size=2,padding='same',name = 'p4')(c4)
    
    c5 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c5')(p4)
    u1 = UpSampling1D(size=2,name = 'u1')(c5)
    
    c6 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c6')(u1)
    u2 = UpSampling1D(size=2, name = 'u2')(c6)
    
    c7 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c7')(u2)
    u3 = UpSampling1D(size=2, name = 'u3')(c7)
    
    c8 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c8')(u3)
    u4 = UpSampling1D(size=2, name = 'u4')(c8)
    
    HRF_output = Conv1D(1,3,padding = 'same',activation = 'linear',name = 'output')(u4)
    
    model = Model(fNIRS_input,HRF_output)
    
    return model
def L8_dropout_arch():
    fNIRS_input = Input(shape = (512,1))
    
    c1 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c1')(fNIRS_input)
    p1 = MaxPooling1D(pool_size=2, padding='same',name = 'p1')(c1)
    d1 = Dropout(0.1)(p1)
    
    c2 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c2')(d1)
    p2 = MaxPooling1D(pool_size=2,padding='same',name = 'p2')(c2)
    d2 = Dropout(0.1)(p2)
    
    c3 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c3')(d2)
    p3 = MaxPooling1D(pool_size=2, padding='same',name = 'p3')(c3)
    d3 = Dropout(0.1)(p3)
    
    c4 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c4')(d3)
    p4 = MaxPooling1D(pool_size=2,padding='same',name = 'p4')(c4)
    d4 = Dropout(0.1)(p4)
    
    c5 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c5')(d4)
    u1 = UpSampling1D(size=2,name = 'u1')(c5)
    d5 = Dropout(0.1)(u1)
    
    c6 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c6')(d5)
    u2 = UpSampling1D(size=2, name = 'u2')(c6)
    d6 = Dropout(0.1)(u2)
    
    c7 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c7')(d6)
    u3 = UpSampling1D(size=2, name = 'u3')(c7)
    d7 = Dropout(0.1)(u3)
    
    c8 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c8')(d7)
    u4 = UpSampling1D(size=2, name = 'u4')(c8)
    d8 = Dropout(0.1)(u4)
    
    HRF_output = Conv1D(1,3,padding = 'same',activation = 'linear',name = 'output')(d8)
    
    model = Model(fNIRS_input,HRF_output)
    
    return model

# %%
model = ['4layers','4layers+dropout','8layers','8layers+dropout']
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
X_real = np.concatenate((X_real_HbO,X_real_HbR),axis = 0)
X = X*1000000
Y = Y*1000000
X_real = X_real*1000000
# %%

SampleSize = X.shape[0]

n_train = np.int16(SampleSize*0.8)

index = np.random.permutation(SampleSize)
X_train = X[index[0:n_train],:]
Y_train = Y[index[0:n_train],:]
X_val = X[index[n_train:],:]
Y_val = Y[index[n_train:],:]

#X_train = preprocessing.normalize(X_train)


X_train = X_train[:,:,np.newaxis]
X_val = X_val[:,:,np.newaxis]
Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]
X_test = X_real[:,:,np.newaxis]

# %%
for model_name in model:
    
    print('Model:', model_name)
    if model_name == '4layers':
        network = L4_arch()
    elif model_name == '4layers+dropout':
        network = L4_dropout_arch()
    elif model_name == '8layers':
        network = L8_arch()
    elif model_name == '8layers+dropout':
        network = L8_dropout_arch()
#    elif model_name == 'try':
#        network = try_arch()
    network.summary()
    hdf5_filepath = "networks\\" + model_name+".hdf5"
    save_model = ModelCheckpoint(hdf5_filepath,monitor='val_loss',save_best_only=True,mode = 'min')
    learning_rate = 0.001
    opt = optimizers.Adam(lr = learning_rate)
    network.compile(loss = 'mean_squared_error', optimizer = opt)
    def step_decay(epoch):
       initial_lrate = 0.0000000001
       drop = 0.5
       epochs_drop = 100.0
       lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
       return lrate
    lr_rate_schedule = LearningRateScheduler(step_decay)
    hist = network.fit_generator(TrainGenerator(np.asarray(X_train),np.asarray(Y_train)),
             steps_per_epoch = 16,
             epochs = 500,
             verbose = 2,
             validation_data = ValGenerator(np.array(X_val),np.array(Y_val)),
             validation_steps=16,
             callbacks = [save_model])
    losses = hist.history
    numpy_loss_history = np.array(losses['loss'])
    val_numpy_loss_history = np.array(losses['val_loss'])
    losspath = "Processed_data\\loss_" + model_name+".txt"
    val_losspath = "Processed_data\\val_loss_" + model_name+".txt"
    np.savetxt(losspath, numpy_loss_history, delimiter=",")
    np.savetxt(val_losspath, val_numpy_loss_history, delimiter=",")
    
    plt.figure()
    vl, = plt.plot(losses['val_loss'],'r')
    tl, = plt.plot(losses['loss'],'b')
    plt.legend([tl,vl],['training loss', 'validation loss'],)
    figurepath = "figures\\loss_" + model_name+".png"
    plt.savefig(figurepath, transparent=True)
    
    network.load_weights(hdf5_filepath)
    Y_test = network.predict(X_test)
    savefilepath = "Processed_data\\Hb_NN_" + model_name+".mat"
    scipy.io.savemat(savefilepath,{'Y_test': Y_test})