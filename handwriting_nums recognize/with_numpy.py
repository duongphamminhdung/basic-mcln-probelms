import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import platform
import warnings

#import mnist dataset
if platform.system() == 'Windows':
    data = pd.read_csv("/mnt/d/DocumentD/GitHub/DATA SETS/mnist/mnist_train.csv")
else:
    data=pd.read_csv('/Users/duongphamminhdung/Documents/GitHub/DATA SETS/mnist/mnist_train.csv')
mnist = data.values
label = mnist[:, 0]
digits = mnist[:, 1:]

X_train = digits[:60000]
Y_train = label[:60000]

def normalize(X):
    X = X / 255
    X -= np.mean(X, axis = 0)
    return X
def find_target(Y):
    """
    tạo one-hot vector 
    """
    Y = Y.reshape(len(Y))
    s = np.zeros((len(Y), 10))
    s[np.arange(Y.size), Y] = 1
    return s
def softmax(Z):
    """
    Tỷ lệ rơi vào các class
    Z.shape = (N_class, 1)
    a_i = e^(z_i)/sum(e^Z)
    """
    ez = np.exp(Z) 
    A = ez / np.sum(ez, axis = 1)[:, None] #dãy tỉ lệ rơi vào class 
    return A
def activate_func(Z):
    #sigmoid
    Z = 1/(np.exp(-Z)+1)
    return Z
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))
