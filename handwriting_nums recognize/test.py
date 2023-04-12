import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from datetime import datetime


training_set = torchvision.datasets.MNIST('./data',download=True,train=True,transform=transforms.ToTensor())
validation_set = torchvision.datasets.MNIST('./data',download=True,train=False,transform=transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(training_set,batch_size=batch,shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch,shuffle=False)

for inputs, targets in training_loader:
    print(inputs.shape, targets.shape)
    break