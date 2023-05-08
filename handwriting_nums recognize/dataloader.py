import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import time
from tqdm import tqdm
import glob
import torchvision
import torch.multiprocessing as mp

# mp.set_start_method('spawn')


path = '/Users/duongphamminhdung/Documents/GitHub/basic mcln probelms/handwriting_nums recognize/data/'

# for i in range(10):
#     folder_name = str(i)
#     folder_path = os.path.join(path, folder_name)
    
#     files = os.listdir(folder_path)
    
# file = os.path.join(path, '5', '5_0.png')
# image = cv2.imread(file, -1)
# print(type(image))
# print(image.shape)

class DatasetLoader(Dataset):
    # def make_list(self, ls):
    #     l_ = []
    #     batch = self.batch
    #     for j in range(0, len(ls), batch):
    #         a = []
    #         for k in range(batch):
    #             if j*batch + k >= len(ls):
    #                 break                
    #             a.append(ls[j*batch + k])
    #         l_.append(a)
    #     return l_
    def __init__(self, path_dataset, batch):
        self.batch = batch
        self.path = path_dataset
        self.list_all_file = glob.glob(os.path.join(self.path, "*", '*.png'))
        
        # print(self.list_all_file)
        # time.sleep(5)
        

    def __len__(self):
        return len(self.list_all_file)

    def __getitem__(self, index):
        image = []
        label = []
        # for j in range(batch):
        #     temp_path = (self.list_all_file[index*batch+j])
        #     img = cv2.imread(temp_path, -1)
        #     lab = os.path.basename(temp_path)[0]
        #     image.append(img)
        #     label.append(int(lab))
        
        # batch = self.batch
        
        # for i in range(batch):
        # # for img_path in self.list_all_file[index*batch : batch*(index+1)]:
        #     img_path = self.list_all_file[index*batch + i]
        #     img = cv2.imread(img_path, -1)
        #     lab = os.path.basename(img_path)[0]
        #     try:
        #         image.append(img)
        #         label.append(int(lab))
        #     except ValueError:
        #         print("eroor")
        #         pass
        img_path = self.list_all_file[index]
        label = os.path.basename(img_path)[0]
        image = cv2.imread(img_path, -1)
        try:
            image = torch.FloatTensor(image)
            image = image.unsqueeze(0)
            label = torch.tensor(int(label))
        except ValueError:
            print("eroor")
            pass
        # image = torch.FloatTensor(image)
        # label = torch.Tensor(label)
        
        return image, label

#   print(image.shape)
if __name__ == "__main__":
    path = '/Users/duongphamminhdung/Documents/GitHub/basic mcln probelms/handwriting_nums recognize/data/'

    dataset = DatasetLoader(path, 4)
    print(len(dataset))
    starttime = time.time()
    # for i, j in tqdm(dataset):
    #     pass
    # print(f'{time.time() - starttime:.2f}s')
    # for image, label  in dataset:

    training_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=4,
                                                shuffle=True,
                                                num_workers=4)
    for data, label in tqdm(training_loader):
        # print(data.shape)
        pass