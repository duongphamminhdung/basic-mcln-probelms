
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

transform = transforms.Compose(
    [transforms.ToTensor(),])

# Store separate training and validations splits in ./data
training_set = torchvision.datasets.MNIST('./data',
    download=True,
    train=True,
    transform=transform)
print(training_set)
validation_set = torchvision.datasets.MNIST('./data',
    download=True,
    train=False,
    transform=transform)

training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)


validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=2)
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.linear1 = nn.Linear(784, 256)
        self.actfunc1 = nn.Sigmoid()
        self.linear2 = nn.Linear(256, 64)
        self.actfunc2 = nn.Sigmoid()
        self.linear3 = nn.Linear(64, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.actfunc1(x)
        x = self.linear2(x)
        x = self.actfunc2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
    

model = MnistModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr = 0.001,
                            momentum = 0.9,)
for i, data in enumerate(training_loader):
    inputs, labels = data
    
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    
    inputs = torch.reshape(inputs, (4, 1, 784))
    outputs = model(inputs)
    
    target = F.one_hot(labels, 10)
    # print(outputs.shape, target.shape)
    print(inputs.shape, labels.shape)
    print(target[:])
    break