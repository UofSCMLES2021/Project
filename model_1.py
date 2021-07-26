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
from sklearn.metrics import mean_squared_error, hinge_loss
import pandas as pd


def graph(col_name):
    data = df.loc[:,["{}".format(col_name)]]
    plt.figure()
    plt.plot(range(len(data)), data, markersize =2)
    plt.title("{}".format(col_name))
    plt.grid(True)
    # plt.tightlayout()
    plt.show()
   
    
def find_coef(col_name, coef_target):
    target = corrMatrix.loc[[col_name],:]
    target_ar = pd.DataFrame.to_numpy(target).T
    target_ar = np.resize(target_ar, (26))
    
    for i in range(target_ar.shape[0]):
        if abs(target_ar[i]) >= coef_target and target_ar[i] < 1:
            print(target.columns[i], ' ', target_ar[i])   

plt.close('all')

#Import and format data
df = pd.read_csv('Inverter Data Set.csv')
df_cols = ['motor speed (min^-1)', 'DC-link voltage (V)',
           'DC-link voltage 1 sampling step before in (V)',
           'DC-link voltage 2 sampling steps before in V',
           'DC-link voltage 3 sampling steps before in V',
           'Phase current of phase a in A',
           'Phase current of phase b in A',
           'Phase current of phase c in A',
           'Phase current of phase a 1 sampling step before in A', 
           'Phase current of phase b 1 sampling step before in A',
           'Phase current of phase c 1 sampling step before in A',
           'Phase current of phase a  sampling steps before in A',
           'Phase current of phase b 2 sampling steps before in A', 
           'Phase current of phase c 2 sampling steps before in A', 
           'Phase current of phase a 3 sampling steps before in A', 
           'Phase current of phase b 3 sampling steps before in A', 
           'Phase current of phase c 3 sampling steps before in A', 
           'Duty cycle of phase a 2 sampling steps before',
           'Duty cycle of phase b 2 sampling steps before', 
           'Duty cycle of phase c 2 sampling steps before', 
           'Duty cycle of phase a 3 sampling steps before', 
           'Duty cycle of phase b 3 sampling steps before', 
           'Duty cycle of phase c 3 sampling steps before', 
           'Measured voltage of phase a 1 sampling step before in V', 
           'Measured voltage of phase b 1 sampling step before in V',
           'Measured voltage of phase c 1 sampling step before in V']
idx = df.columns
df.columns = df_cols
plt.close('all')

#print names of all the column headings
# for cols in df.columns:
#     print(cols)

y=df.loc[:,['Phase current of phase a in A']]
x=df.loc[:,['motor speed (min^-1)']]
z=df.loc[:,['Measured voltage of phase a 1 sampling step before in V']]
x, y, z = x.to_numpy(), y.to_numpy(), z.to_numpy()

#data for the first motor speed
# X = x[(x>0) & (x<1000)] #motor speed
# Y = z[(x>0) & (x<1000)] #phase voltage

# X = np.reshape(X, (X.shape[0],1))
# Y = np.reshape(Y, (Y.shape[0],1))
# print(Y.shape)
#%%Correllation Coefficients
corrMatrix = df.corr(method = 'pearson')
find_coef('Measured voltage of phase a 1 sampling step before in V', .7)



#%%Load in Data
x_values, y_values = np.asarray(df.loc[:, ['Phase current of phase a in A','Phase current of phase a  sampling steps before in A','Phase current of phase a 1 sampling step before in A', 'motor speed (''min^-1)']]), \
                          np.asarray(z)
# x_values, y_values = np.asarray(df.loc[:, ['Phase current of phase b in A','Phase current of phase b 2 sampling steps before in A','Phase current of phase b 1 sampling step before in A', 'motor speed (''min^-1)','Duty cycle of phase a 2 sampling steps before','Duty cycle of phase a 3 sampling steps before']]), \
#                          np.asarray(z)

y_values = y_values.squeeze(axis=1)

#%%Generate learning curves for polynomial model
train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(x_values, y_values, test_size = 0.2)
model = sk.linear_model.LinearRegression()


model.fit(train_x, train_y)
print('Model Score: {}'.format(model.score(test_x, test_y)))
y_pred = model.predict(test_x)
print('Training MSE: {}'.format(mean_squared_error(train_y, model.predict(train_x))))
print('MSE: {}'.format(mean_squared_error(test_y, y_pred)))
plt.figure()
plt.plot(range(len(test_y)), test_y, color = 'b', label = 'real')
plt.plot(range(len(test_y)), y_pred, 'o--', color = 'r', dashes= ((5,8)), linewidth = 1, label = 'prediction')
plt.legend()
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(train_errors, '--', label = 'training error')
# plt.plot(val_errors, '--', label = 'validation error')
# plt.xlabel('number of data points used in training')
# plt.ylabel('mean squared error')
# plt.grid()
# plt.tight_layout()
# plt.legend(framealpha = 1)




#%%

