# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:28:09 2016

@author: aslado
"""

import os
import inspect
import tensorflow
print(os.path.dirname(inspect.getfile(tensorflow)))

# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import numpy as np
import tensorflow as tf
import math

def add_one():
    with tf.Session() as session:
        # (1)
        x = tf.placeholder(tf.float32, [1], name='x') # fed as input below
        y = tf.placeholder(tf.float32, [1], name='y') # fetched as output below
        b = tf.constant(1.0)
        y = x + b # here is our ‘model’: add one to the input.
        x_in = [2] # (2)
        y_final = session.run([y], {x: x_in}) # (3)
    print(y_final) # (4)
add_one()
