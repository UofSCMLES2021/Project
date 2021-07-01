# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:35:10 2021

@author: BPRIDDY
"""

import IPython as IP
IP.get_ipython().magic("reset -sf")

#import modules native to python
import random as random
import sys as sys
import os as os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
from scipy import signal
from scipy.fft import fft, fftfreq
import sklearn as sk
from sklearn import datasets, preprocessing, linear_model, pipeline
import pandas as pd
#EXAMPLE DATA FROM THE CLASS
plt.close('all')
np.random.seed(2)
x = np.random.rand(1)

m = 100 # number of samples
X = 6*np.random.rand(m,1)-3
X_n = 6*np.random.rand(m,1)-5
#y_hat = a*x**2 + b*x + c
Y = 0.5*X**2 + X + 2 + np.random.randn(m,1)
print(type(X))

plt.figure()
plt.plot(X,Y, 'o')

plt.ylim(0,10)
plt.show()


#Import and format data
df = pd.read_csv('Inverter_Data_Set.csv')


df_cols = ['motor speed (min^-1)', 'DC-link voltage (V)', 'DC-link voltage 1 sampling step before in (V)', 'DC-link voltage 2 sampling steps before in V', 'DC-link voltage 3 sampling steps before in V', 'Phase current of phase a in A', 'Phase current of phase b in A', 'Phase current of phase c in A', 'Phase current of phase a 1 sampling step before in A',  'Phase current of phase b 1 sampling step before in A','Phase current of phase c 1 sampling step before in A', 'Phase current of phase a  sampling steps before in A', 'Phase current of phase b 2 sampling steps before in A', 'Phase current of phase c 2 sampling steps before in A', 'Phase current of phase a 3 sampling steps before in A', 'Phase current of phase b 3 sampling steps before in A', 'Phase current of phase c 3 sampling steps before in A', 'Duty cycle of phase a 2 sampling steps before', 'Duty cycle of phase b 2 sampling steps before', 'Duty cycle of phase c 2 sampling steps before', 'Duty cycle of phase a 3 sampling steps before', 'Duty cycle of phase b 3 sampling steps before', 'Duty cycle of phase c 3 sampling steps before', 'Measured voltage of phase a 1 sampling step before in V', 'Measured voltage of phase b 1 sampling step before in V', 'Measured voltage of phase c 1 sampling step before in V']

df.columns = df_cols
plt.close('all')

#print names of all the column headings
for cols in df.columns:
    print(cols)

y=df.loc[:,['Phase current of phase a in A']]
x=df.loc[:,['motor speed (min^-1)']]
z=df.loc[:,['Measured voltage of phase a 1 sampling step before in V']]
x, y, z = x.to_numpy(), y.to_numpy(), z.to_numpy()

#data for the first motor speed
X = x[(x>0) & (x<1000)] #motor speed
Y = z[(x>0) & (x<1000)] #phase voltage

X = np.reshape(X, (X.shape[0],1))
Y = np.reshape(Y, (Y.shape[0],1))
print(Y.shape)
#%%pERFORM THE POLYNOMIAL REGRESSION

# =============================================================================
# SOME RANDOM BULLSHIT
# poly_features = sk.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
# X_poly_sk = poly_features.fit_transform(X)
# =============================================================================


X_model = np.linspace(-3,3)

# =============================================================================
# #plot the features with the linear model
# plt.figure()
# plt.grid(True)
# plt.scatter(X_poly[:, 0],Y, label = 'data for $X$')
# plt.scatter(X_poly[:, 1],Y, label = 'data for $X^2$')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.plot(X_model, Y_X, '--', label = 'linear fit for $X$')
# plt.plot(X_model2, Y_2, '--', label = 'linear fit for $X^2$')
#
#
# plt.legend()
# =============================================================================


#%%Generate learning curves for polynomial model
X_train, X_val, Y_train, Y_val = sk.model_selection.train_test_split(X, Y, test_size = 0.2)
model = sk.pipeline.Pipeline((("poly_features", sk.preprocessing.PolynomialFeatures(degree = 20, include_bias=False)),("lin_reg", sk.linear_model.LinearRegression()),))

# =============================================================================
# pipeline_model = sk.preprocessing.PolynomialFeatures(degree = 20, include_bias=False)
# pipeline_solver =  sk.linear_model.LinearRegression()
#
# model = sk.pipeline.Pipeline(("poly_features", pipeline_model),("lin_reg", pipeline_solver))
# =============================================================================

train_errors, val_errors = [], []
for i in range(1,len(X_train)):
    # print(i)
    model.fit(X_train[:i], Y_train[:i])
    y_train_predict = model.predict(X_train[:i])
    y_val_predict = model.predict(X_val)
    mse_train = sk.metrics.mean_squared_error(Y_train[:i], y_train_predict)
    train_errors.append(mse_train)
   
    mse_val = sk.metrics.mean_squared_error(Y_val, y_val_predict)
    val_errors.append(mse_val)
   
    if i == len(X_train)-1:
        plt.figure()
        plt.scatter(X, Y, s = 2, label = 'data')
        plt.scatter(X_train[:i],Y_train[:i], s = 2, label = 'data in training set')
        plt.scatter(X_val,Y_val, s = 2, marker = 's', label = 'data in validation set')
        y_model = model.predict(np.expand_dims(X_model, axis = 1))
        plt.plot(X_model, y_model, 'r--', label='model')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend(loc=2)
        plt.show()



plt.figure()
plt.plot(train_errors, '--', label = 'training error')
plt.plot(val_errors, '--', label = 'validation error')
plt.xlabel('number of data points used in training')
plt.ylabel('mean squared error')
plt.grid()
plt.tight_layout()
plt.legend(framealpha = 1)




#%%

