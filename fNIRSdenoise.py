#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:53:55 2020

@author: gaoyuanyuan
"""

from keras.models import Model
from keras.layers import Conv1D, Input,MaxPooling1D,UpSampling1D,Concatenate,Lambda
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
from keras import optimizers
import scipy.io
from sklearn import preprocessing
import keras.backend as K
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
    fNIRS_input = Input(shape = (1024,1))
    input1 = Lambda(lambda x: x[:,0:512, :])(fNIRS_input)
    input2 = Lambda(lambda x: x[:,512:, :])(fNIRS_input)
    
    conv1 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c1')
    c1_1 = conv1(input1)
    p1_1 = MaxPooling1D(pool_size=2, padding='same',name = 'p1_1')(c1_1)
    c1_2 = conv1(input2)
    p1_2 = MaxPooling1D(pool_size=2, padding='same',name = 'p1_2')(c1_2)
    
    conv2 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c2')
    c2_1 = conv2(p1_1)
    p2_1 = MaxPooling1D(pool_size=2,padding='same',name = 'p2_1')(c2_1)
    c2_2 = conv2(p1_2)
    p2_2 = MaxPooling1D(pool_size=2,padding='same',name = 'p2_2')(c2_2)
    
    conv3 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c3')
    c3_1 = conv3(p2_1)
    u1_1 = UpSampling1D(size=2,name = 'u1_1')(c3_1)
    c3_2 = conv3(p2_2)
    u1_2 = UpSampling1D(size=2,name = 'u1_2')(c3_2)
    
    conv4 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c4')
    c4_1 = conv4(u1_1)
    u2_1 = UpSampling1D(size=2, name = 'u2_1')(c4_1)
    c4_2 = conv4(u1_2)
    u2_2 = UpSampling1D(size=2, name = 'u2_2')(c4_2)
    
    conv5 = Conv1D(1,3,padding = 'same',activation = 'linear',name = 'output')
    HRF_output_1 = conv5(u2_1)
    HRF_output_2 = conv5(u2_2)
#    HRF_output = HRF_output_1
    HRF_output = Concatenate(axis = 1)([HRF_output_1,HRF_output_2])
    
    model = Model(fNIRS_input,HRF_output)
    
    return model

def L8_arch():
    fNIRS_input = Input(shape = (1024,1))
    
    c1 = Conv1D(32,11,padding = 'same',activation = 'relu',name = 'c1')(fNIRS_input)
    p1 = MaxPooling1D(pool_size=2, padding='same',name = 'p1')(c1)
    
    c2 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c2')(p1)
    p2 = MaxPooling1D(pool_size=2,padding='same',name = 'p2')(c2)
    
    c3 = Conv1D(32,3,padding = 'same',activation = 'relu',name = 'c3')(p2)
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


# %%
model = ['4layers','8layers']
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
# %%
SampleSize = int(X.shape[0]/2)
X_HbO = X[0:SampleSize,:]
X_HbR = X[SampleSize:,:]
Y_HbO = Y[0:SampleSize,:]
Y_HbR = Y[SampleSize:,:]

n_train = np.int16(SampleSize*0.8)

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
    elif model_name == '8layers':
        network = L8_arch()
#    elif model_name == 'temp':
#        network = temp()
    network.summary()
    hdf5_filepath = "networks\\" + model_name+".hdf5"
    save_model = ModelCheckpoint(hdf5_filepath,monitor='val_loss',save_best_only=True,mode = 'min')
    learning_rate = 0.00001
    opt = optimizers.Adam(lr = learning_rate)
    def SNR(y_true, y_pred):
        return K.mean(y_pred)/(K.std(y_pred)+0.00000001)
    def mse_and_SNR(y_true, y_pred):
        mse = K.mean(K.sum(K.square(y_true-y_pred)))
        SNR = K.abs(K.mean(y_pred)/(K.std(y_pred)+0.00000001))
        
        return mse - SNR
#    network.compile(loss = 'mean_squared_error',
#                optimizer = opt, metrics=[SNR])
    network.compile(loss = mse_and_SNR,optimizer = opt)
    def step_decay(epoch):
       initial_lrate = 0.000001
       drop = 0.5
       epochs_drop = 100.0
       lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
       return lrate
    lr_rate_schedule = LearningRateScheduler(step_decay)
    hist = network.fit_generator(TrainGenerator(np.asarray(X_train),np.asarray(Y_train)),
             steps_per_epoch = 16,
             epochs = 1000,
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