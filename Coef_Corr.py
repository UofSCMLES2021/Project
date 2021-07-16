#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:24:36 2021

@author: bradenpriddy
"""

import sys
import csv
import argparse
import sqlite3
import IPython as IP
IP.get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from datetime import datetime, timedelta
import json
import IPython
import pandas as pd
import seaborn as sn

df = pd.read_csv('Inverter Data Set.csv')
plt.close('all')
df_cols = ['motor speed (min^-1)', 'DC-link voltage (V)', 'DC-link voltage 1 sampling step before in (V)', 'DC-link voltage 2 sampling steps before in V', 'DC-link voltage 3 sampling steps before in V', 'Phase current of phase a in A', 'Phase current of phase b in A', 'Phase current of phase c in A', 'Phase current of phase a 1 sampling step before in A',  'Phase current of phase b 1 sampling step before in A','Phase current of phase c 1 sampling step before in A', 'Phase current of phase a  sampling steps before in A', 'Phase current of phase b 2 sampling steps before in A', 'Phase current of phase c 2 sampling steps before in A', 'Phase current of phase a 3 sampling steps before in A', 'Phase current of phase b 3 sampling steps before in A', 'Phase current of phase c 3 sampling steps before in A', 'Duty cycle of phase a 2 sampling steps before', 'Duty cycle of phase b 2 sampling steps before', 'Duty cycle of phase c 2 sampling steps before', 'Duty cycle of phase a 3 sampling steps before', 'Duty cycle of phase b 3 sampling steps before', 'Duty cycle of phase c 3 sampling steps before', 'Measured voltage of phase a 1 sampling step before in V', 'Measured voltage of phase b 1 sampling step before in V', 'Measured voltage of phase c 1 sampling step before in V']
df.columns = df_cols


#print names of all the column headings
#for cols in df.columns:
#    print(cols)
    


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
            print(target.columns[i], '   ', target_ar[i])   
    

#graph('motor speed (min^-1)')

#CORRELATION MATRIX
corrMatrix = df.corr(method = 'pearson')
#sn.heatmap(corrMatrix, annot=True)
plt.show()


find_coef('Measured voltage of phase a 1 sampling step before in V', .7)

