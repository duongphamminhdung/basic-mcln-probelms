import cv2
import torch
import glob
import os
from tqdm import tqdm

path = '/Users/duongphamminhdung/Documents/GitHub/basic mcln probelms/handwriting_nums recognize/data/'
ls = glob.glob(os.path.join(path, "*", '*.png'))
batch = 4
# for j in range(0, len(l), batch):
#     a = []
#     for k in range(batch):
#         if j*batch + k >= len(l):
#             print(len(ls))
#             break
        
#         a.append(l[j*batch + k])
#     ls.append(a)

def get_item(index):
    if (index+1)*batch >= len(ls):
        return False
    val = ls[index*batch : (index+1)*batch]
    image = []
    for image_path in val:
        # print(image_path)
        img = cv2.imread(image_path, -1)
        image.append(torch.Tensor(img))
        
    return image

def my_generator(ls, batch):

    # initialize counter
    value = 0
    # loop until counter is less than n
    for value in range(len(ls)//batch):

        try:
            res = get_item(value)
        except IndexError:
            break
        
        yield res

        # increment the counter

# iterate over the generator object produced by my_generator
for value in tqdm(my_generator(ls, batch), total=len(ls)//batch):

    # print each value produced by generator
    pass
print(len(ls))
print(len(my_generator(ls, batch)))
