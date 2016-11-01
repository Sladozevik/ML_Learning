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

def computeCost(X, y, theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
    
# add columnt of 1's in the fornt of the data set

data.insert(0,'Ones', 1)

# set X as training data and y target variable

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# Convert to data frames to numpy matrices

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# to see shape of matrix

print('X', X.shape,'theta', theta.shape,'y', y.shape)
print(computeCost(X,y,theta))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(parameters):
        error = (X*theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X))*np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X,y,theta)
        
    return theta, cost

# adding variables

alpha = 0.01
iters = 1000

# perform gradient descent

g, cost = gradientDescent(X, y, theta, alpha, iters)
g
computeCost(X,y,g)

#viewing results

x = np.linspace(data.Population.min(), data.Population.max(), 100)  
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  

# error curve

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
