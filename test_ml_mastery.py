# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:57:20 2016

@author: aslado
"""

import numpy
import pandas

myarray = numpy.array([[1,2,3],[4,5,6]])
rownames = ['a','b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print mydataframe

# Load CVS using Pandas from URL
from pandas import read_csv
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'aga', 'class']
data = read_csv(url, names=names)
print 'data.shape:', data.shape