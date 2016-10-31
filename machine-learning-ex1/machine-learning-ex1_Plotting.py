# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:44:04 2016

@author: Ante Sladojevic
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

# import data and visualizing it

path = os.getcwd() + '\ex1\ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profit'])
print(data.head()) # return of datas first rows
print(data.describe()) #  calculate some basic statistics on a data set.

data.plot(kind='scatter', x='Population', y='Profit', figsize=(10,10))

# Simple Linear Regression

def computeCost(X, y, thera):
    inner = np.power((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
    
    