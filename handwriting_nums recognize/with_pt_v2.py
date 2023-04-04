import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from datetime import datetime


# Define the model
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        
        #CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size = (3, 3), stride=1, padding=1)
        self.act1 = nn.Sigmoid() 
        #in 1x28x28 out 3x28x28
        
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size= (3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        #in 3x28x28 out 9x14x14
        
        self.flat = nn.Flatten()
        #int 9x14x14 out 
        #linear layers
        self.linear1 = nn.Linear(9*14*14, 256)
        self.actfunc1 = nn.Sigmoid()
        self.linear2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = (self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        # import ipdb;ipdb.set_trace()
        x = self.flat(x)
        x = self.linear2(self.actfunc1(self.linear1(x)))
        x = self.softmax(x)
        
        return x

# Load the data
batch = 16
# Store separate training and validations splits in ./data
training_set = torchvision.datasets.MNIST('./data',download=True,train=True,transform=transforms.ToTensor())
validation_set = torchvision.datasets.MNIST('./data',download=True,train=False,transform=transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(training_set,batch_size=batch,shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=batch,shuffle=True)

# Define the model and loss function
model = MnistModel()
# model.cuda()
criterion = nn.CrossEntropyLoss()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
NUM_EPOCHS = 50
acc = []
losses = []
best_acc = 80
# Train the model
for epoch in range(NUM_EPOCHS):
    # Train the model for one epoch
    train_loss = 0.0
    train_correct = 0
    model.train()
    for inputs, targets in training_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()
    train_loss /= len(training_loader.dataset)
    train_accuracy = train_correct / len(training_loader.dataset)

    # Test the model
    test_loss = 0.0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in validation_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

    test_loss /= len(validation_loader.dataset)
    test_accuracy = test_correct / len(validation_loader.dataset)
    acc.append(test_accuracy)
    losses.append(test_loss)
    # Print progress
    if acc[-1] > best_acc:
        model_path = 'best_model_{}'.format(timestamp)
        torch.save(model.state_dict(), "weights/"+model_path)
        
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}'.format(
        epoch+1, NUM_EPOCHS, train_loss, train_accuracy*100, test_loss, test_accuracy*100))
    
    plt.grid()
plt.plot([i for i in range(NUM_EPOCHS)], losses, label="loss")
plt.plot([i for i in range(NUM_EPOCHS)], acc, label="accuracy")
plt.legend(loc = 2)
plt.show()
