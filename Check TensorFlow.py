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