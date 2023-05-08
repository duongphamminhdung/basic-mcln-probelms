import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import platform
import cv2
import os

#import mnist dataset
if platform.system() == 'Windows':
    data = pd.read_csv("/mnt/d/DocumentD/GitHub/DATA SETS/mnist/mnist_train.csv")
else:
    data=pd.read_csv('/Users/duongphamminhdung/Documents/GitHub/DATA SETS/mnist/mnist_train.csv')
mnist = data.values
label = mnist[:, 0]
digits = mnist[:, 1:]

path = '/Users/duongphamminhdung/Documents/GitHub/basic mcln probelms/handwriting_nums recognize/data/'
for i in range(len(label)):
    im = digits[i].reshape(28, 28)
    print('hehe')
    
    folder_name = str(label[i])
    folder_path = os.path.join(path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, f'{folder_name}_{i}.png')
    cv2.imwrite(save_path, im)
    print(save_path)
