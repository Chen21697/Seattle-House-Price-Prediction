#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:43:11 2020

@author: yuwenchen
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalization(x):
    return (x - min(x)) / (max(x) - min(x))

def val_normalization(x):
    global maxMinData
    return (x - maxMinData.iloc[1]) / (maxMinData.iloc[0] - maxMinData.iloc[1])

def findMaxMin(x):
    return x.max(axis=0), x.min(axis=0)

def BGD(x, x_grad, lr):
    '''
    Batch Gradient Descent
    
    required input:
        x: current parameter
        grad_x: the gradient of current parameter
        lr: fixed learniing rate
    output:
        updated parameter
    '''
    return x - lr*x_grad


def Adagrad(x, x_grad, lr, pre_lr):
    '''
    Adagrad Optimizer
    
    required input:
        x: current parameter
        grad_x: the gradient of current parameter
        lr: fixed learniing rate
        pre_lr: last learning rate for each parameters
    output:
        updated parameter, new learning rate 
    '''
    lr_x = pre_lr + x_grad**2
    return x - lr/np.sqrt(lr_x)*x_grad, lr_x

w_mt, w_vt, b_mt, b_vt = 0, 0, 0, 0

def adam(x, x_grad, lr, curIter, mt, vt):
    '''
    Adam Optimizer
    
    required input:
        x: current parameter
        grad_x: the gradient of current parameter
        lr: fixed learniing rate
        curIter: current iteration number
        mt: momentum term
        vt: RMSprop term 
    '''
    b1 = 0.9
    b2 = 0.999
    e = 0.00000001
    
    mt = b1*mt + (1-b1)*x_grad
    vt = b2*vt + (1-b2)*(np.power(x_grad, 2))
    
    m_hat = mt/(1-np.power(b1, curIter+1))
    v_hat = vt/(1-np.power(b2, curIter+1))
    
    new_g = x - lr*m_hat/(np.sqrt(v_hat) + e)
    return new_g, mt, vt
#%%
trainingSet = pd.DataFrame(pd.read_csv('PA1_train.csv'))
valSet = pd.DataFrame(pd.read_csv('PA1_dev.csv'))

temp = trainingSet["date"]
temp2 = temp.str.split("/", n=2, expand = True)
temp2.astype(int)

trainingSet["month"] = temp2[0]
trainingSet["day"] = temp2[1]
trainingSet["year"] = temp2[2]

trainingSet['month'] = trainingSet['month'].astype(int)
trainingSet['day'] = trainingSet['day'].astype(int)
trainingSet['year'] = trainingSet['year'].astype(int)

temp_v = valSet["date"]
temp2_v = temp_v.str.split("/", n=2, expand = True)
temp2_v.astype(int)

valSet["month"] = temp2_v[0]
valSet["day"] = temp2_v[1]
valSet["year"] = temp2_v[2]

valSet['month'] = valSet['month'].astype(int)
valSet['day'] = valSet['day'].astype(int)
valSet['year'] = valSet['year'].astype(int)

train_numFea = trainingSet[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 'lat', 'long', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]
train_catFea = trainingSet[['waterfront', 'yr_built', 'yr_renovated', 'zipcode', 'month', 'day', 'year']]

vali_numFea = valSet[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view','condition', 'grade', 'sqft_above', 'lat', 'long', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]
vali_catFea = valSet[['waterfront', 'yr_built', 'yr_renovated', 'zipcode', 'month', 'day', 'year']]
#%%
train_yr_built = pd.get_dummies(train_catFea['yr_built'],prefix='yr_built')
# transform yr_renovated column to 0 or 1
train_catFea['yr_renovated'] = train_catFea['yr_renovated'].apply(lambda x: 0 if x==0 else 1)
train_yr_renovated = pd.get_dummies(train_catFea['yr_renovated'],prefix='yr_renovated')

train_zipcode = pd.get_dummies(train_catFea['zipcode'],prefix='zipcode')
train_month = pd.get_dummies(train_catFea['month'],prefix='month')
train_day = pd.get_dummies(train_catFea['day'],prefix='day')
train_year = pd.get_dummies(train_catFea['year'],prefix='year')

concat_train_cat = pd.concat([train_catFea, train_yr_built, train_yr_renovated, train_zipcode, train_month, train_day, train_year], axis=1)
train_catFea = concat_train_cat.drop(columns=['yr_built', 'yr_renovated', 'zipcode', 'month', 'day', 'year'])
#%%
vali_yr_built = pd.get_dummies(vali_catFea['yr_built'],prefix='yr_built')
# transform yr_renovated column to 0 or 1
vali_catFea['yr_renovated'] = vali_catFea['yr_renovated'].apply(lambda x: 0 if x==0 else 1)
vali_yr_renovated = pd.get_dummies(vali_catFea['yr_renovated'],prefix='yr_renovated')

vali_zipcode = pd.get_dummies(vali_catFea['zipcode'],prefix='zipcode')
vali_month = pd.get_dummies(vali_catFea['month'],prefix='month')
vali_day = pd.get_dummies(vali_catFea['day'],prefix='day')
vali_year = pd.get_dummies(vali_catFea['year'],prefix='year')

concat_vali_cat = pd.concat([vali_catFea, vali_yr_built, vali_yr_renovated, vali_zipcode, vali_month, vali_day, vali_year], axis=1)
vali_catFea = concat_vali_cat.drop(columns=['yr_built', 'yr_renovated', 'zipcode', 'month', 'day', 'year'])

#%%
maxMinData = train_numFea.apply(findMaxMin)
norm_train_numFea = train_numFea.apply(normalization)
norm_vali_numFea = val_normalization(vali_numFea)

#%%
train_x = pd.concat([norm_train_numFea, train_catFea], axis=1)
test_x = pd.concat([norm_vali_numFea, vali_catFea], axis=1)
train_y = trainingSet[['price']]
test_y = valSet[['price']]

print("Data preprocessing was done!!!!")
#%%
lenFeature = len(train_x.columns)
sampleNum = len(train_x)
validNum = len(test_x)

w = np.full(lenFeature, 0.1)
b = np.array([0.1])

w_grad = np.zeros(lenFeature)
b_grad = np.array([0])

lr = 0.001

# for Adagrad
lr_w = 0
lr_b = 0

epoch = 20000
loss_record = []
b_record = []
valid_loss_record = []

x = train_x.to_numpy() # transform all data in to numpy form
temp_x = test_x.to_numpy() # transform all validation set in to numpy form
y_head = train_y.to_numpy().reshape(sampleNum) 
temp_y_head = test_y.to_numpy().reshape(validNum) 

for i in range(epoch):
    y = np.sum(x*w+b, axis=1)
    loss = ((np.power((y_head - y), 2)).sum())/sampleNum # caculate loss for all samples
    loss_record.append(loss)
    
    temp_y = np.sum(temp_x*w+b, axis=1)
    valid_loss = ((np.power((temp_y_head - temp_y), 2)).sum())/validNum
    valid_loss_record.append(valid_loss)
    
    #compute ∂L/∂w
    w_grad = ((2.0*(y_head - y))@(-x))/sampleNum
    #compute ∂L/∂b
    b_grad = np.sum((2.0*(y_head - y))*(-1))/sampleNum
    
    #Use no optimizer to update
    w = BGD(w, w_grad, lr)
    b = BGD(b, b_grad, lr)
    
    #Use Adagrad as optimizer to update
    #w, lr_b = Adagrad(w, w_grad, lr, lr_b)
    #b, lr_w = Adagrad(b, w_grad, lr, lr_b)
    
    
    #Use Adam as optimizer to update
    #w, w_mt, w_vt = adam(w, w_grad, lr, i, w_mt, w_vt)
    #b, b_mt, b_vt = adam(b, b_grad, lr, i, b_mt, b_vt)
    
    
    if i%100 == 0: # tracing loss while training
        print("epoch:", i, " loss:", loss, "valid_loss", valid_loss, 'norm', np.linalg.norm(w_grad))
        
    if (np.linalg.norm(w_grad, ord=2) < 0.3):
        print("epoch:", i, " loss:", loss, "valid_loss", valid_loss, 'norm', np.linalg.norm(w_grad))
        break