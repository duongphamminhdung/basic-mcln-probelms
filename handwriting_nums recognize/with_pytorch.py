# PyTorch model and training necessities
# Image display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')

from datetime import datetime

batch = 100

transform = transforms.Compose(
    [transforms.ToTensor(),
])
# Store separate training and validations splits in ./data
training_set = torchvision.datasets.MNIST('./data',
    download=True,
    train=True,
    transform=transform)
validation_set = torchvision.datasets.MNIST('./data',
    download=True,
    train=False,
    transform=transform)

training_loader = torch.utils.data.DataLoader(training_set,
                                              batch_size=batch,
                                              shuffle=True,
                                              num_workers=2)


validation_loader = torch.utils.data.DataLoader(validation_set,
                                                batch_size=batch,
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        # x = self.actfunc1(x)
        x = self.linear2(x)
        # x = self.actfunc2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
    

model = MnistModel()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr = 0.001,
                            momentum = 0.9,)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        inputs = torch.reshape(inputs, (batch, 784))
        outputs = model(inputs)
        
        # Compute the loss and its gradients
        # target = F.one_hot(labels, 10)
        target = labels
        loss = loss_fn(outputs, target)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

epoch_number = 0
EPOCHS = 10
best_vloss = 1_000_000.
accs = []
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    
    # We don't need gradients on to do reporting
    model.train(False)
    
    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = torch.reshape(vinputs, (batch, 784))
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

        predict_labels = torch.argmax(voutputs, dim=1)
        accs.append('accuracy: {}'.format((predict_labels==vlabels).sum()))
        # import ipdb; ipdb.set_trace()
    avg_vloss = running_vloss / (i + 1)
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss and epoch==10:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), "weights/"+model_path)
    
    epoch_number += 1
    print('max accuracy: {}'.format(max(accs)))
