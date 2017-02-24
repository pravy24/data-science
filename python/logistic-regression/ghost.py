# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 20:45:01 2017

@author: Z003EY2A
"""

import numpy as npy
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression as logreg

if __name__ == "__main__":
    
    data = npy.loadtxt("D:\\IC019134\\Learnings\\Data Science\\Kaggle\\Ghost CC\\train.csv", 
                           delimiter=",",  skiprows=1, usecols=(1,2,3,4), ndmin=2).reshape(371, 4)
    
    