import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


# Define the model
class LinearModel(nn.Module):
    def __init__(self, input_size=784, output_size=10):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.actfunc1 = nn.Sigmoid()
        self.linear3 = nn.Linear(256, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.actfunc1(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

# Load the data
train_data = MNIST(root='./data', train=True, transform=ToTensor())
test_data = MNIST(root='./data', train=False, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define the model and loss function
model = LinearModel(input_size=28*28, output_size=10)
# model.cuda()
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
NUM_EPOCHS = 50

# Train the model
for epoch in range(NUM_EPOCHS):
    # Train the model for one epoch
    train_loss = 0.0
    train_correct = 0
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    # Test the model
    test_loss = 0.0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.view(inputs.shape[0], -1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)

    # Print progress
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(
        epoch+1, NUM_EPOCHS, train_loss, train_accuracy, test_loss, test_accuracy))