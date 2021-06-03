# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:20:09 2021

@author: BPRIDDY
"""

import sys
import csv
import argparse
import sqlite3
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from datetime import datetime, timedelta
import json
import IPython
import pandas as pd

df = pd.read_csv('Inverter_Data_Set.csv')


df_cols = ['motor speed (min^-1)', 'DC-link voltage (V)', 'DC-link voltage 1 sampling step before in (V)', 'DC-link voltage 2 sampling steps before in V', 'DC-link voltage 3 sampling steps before in V', 'Phase current of phase a in A', 'Phase current of phase b in A', 'Phase current of phase c in A', 'Phase current of phase a 1 sampling step before in A',  'Phase current of phase b 1 sampling step before in A','Phase current of phase c 1 sampling step before in A', 'Phase current of phase a  sampling steps before in A', 'Phase current of phase b 2 sampling steps before in A', 'Phase current of phase c 2 sampling steps before in A', 'Phase current of phase a 3 sampling steps before in A', 'Phase current of phase b 3 sampling steps before in A', 'Phase current of phase c 3 sampling steps before in A', 'Duty cycle of phase a 2 sampling steps before', 'Duty cycle of phase b 2 sampling steps before', 'Duty cycle of phase c 2 sampling steps before', 'Duty cycle of phase a 3 sampling steps before', 'Duty cycle of phase b 3 sampling steps before', 'Duty cycle of phase c 3 sampling steps before', 'Measured voltage of phase a 1 sampling step before in V', 'Measured voltage of phase b 1 sampling step before in V', 'Measured voltage of phase c 1 sampling step before in V']

df.columns = df_cols
plt.close('all')

#print names of all the column headings
for cols in df.columns:
    print(cols)
    
    
def graph(col_name):
    data = df.loc[:,["{}".format(col_name)]]
    plt.figure()
    plt.plot(range(len(data)), data, markersize =2)
    plt.title("{}".format(col_name))
    plt.grid(True)
    # plt.tightlayout()
    plt.show()
    
graph(df_cols[0])
graph(df_cols[1])




data = df.loc[:,['DC-link voltage (V)']]

y=df.loc[:,['DC-link voltage (V)']]
x=df.loc[:,['motor speed (min^-1)']]
plt.figure()
plt.plot(x, y, 'o', markersize =2)
plt.xlabel('motor speed (min^-1)')
plt.ylabel('DC-link voltage (V)')
plt.grid(True)
plt.show()