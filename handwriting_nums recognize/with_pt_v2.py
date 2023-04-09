import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
from datetime import datetime
import os

torch.seed()
# Define the model
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        
        #CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size = (3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU() 
        #in 1x28x28 out 3x28x28
        
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size= (3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        #in 3x28x28 out 9x14x14
        
        self.flat = nn.Flatten()
        #int 1x1764 out


        #linear layers
        self.linear1 = nn.Linear(9*14*14, 256)
        self.actfunc1 = nn.Sigmoid()
        self.linear2 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = (self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        # import ipdb;ipdb.set_trace()
        x = (self.flat(x))
        x = self.linear2(self.actfunc1(self.linear1(x)))
        x = self.softmax(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on", device, end=' ')
torch.manual_seed(256)

batch = 16
# Store separate training and validations splits in ./data
training_set = torchvision.datasets.MNIST('./data',download=True,train=True,transform=transforms.ToTensor())
validation_set = torchvision.datasets.MNIST('./data',download=True,train=False,transform=transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(training_set,batch_size=batch,shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=1,shuffle=True)

# Define the model and loss function
model = MnistModel().to(device)
# model.cuda()
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Define the optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.01) 
# optimizer = optim.RMSprop(model.parameters(), lr=0.01)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
print('loss_func:', criterion, "optimizer:", optimizer)

NUM_EPOCHS = 200
acc = []
losses = []
best_acc = 98
# Train the model
try:
  for epoch in range(NUM_EPOCHS):
      # Train the model for one epoch
      train_loss = 0.0
      train_correct = 0
      model.train()
      for inputs, targets in training_loader:
          inputs, targets = inputs.to(device), targets.to(device)
          outputs = model(inputs)
          targets_one_hot = F.one_hot(targets, 10).to(torch.float32)
          loss = criterion(outputs, targets_one_hot)

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
              inputs, targets = inputs.to(device), targets.to(device)
              outputs = model(inputs)
              targets_one_hot = F.one_hot(targets, 10)
              loss = criterion(outputs, targets_one_hot)

              test_loss += loss.item()
              test_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()

      test_loss /= len(validation_loader.dataset)
      test_accuracy = test_correct / len(validation_loader.dataset) *100

      #save loss and accuracy
      acc.append(test_accuracy)
      losses.append(test_loss)

      #Save the best model
      if acc[-1] > best_acc:
          model_path = 'best_model_{}'.format(timestamp)
          weight_folder = 'weights'
          os.makedirs(weight_folder, exist_ok=True)
          best_weight_path = os.path.join(weight_folder, model_path)
          torch.save(model.state_dict(), best_weight_path)

      # Print progress
      print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}'.format(
          epoch+1, NUM_EPOCHS, train_loss, train_accuracy*100, test_loss, test_accuracy))
      if acc[-1] > 99 or epoch > 500:
        break
  plt.grid()
  plt.plot([i for i in range(len(acc))], acc, label="accuracy")
  plt.show()
except KeyboardInterrupt:
  plt.grid()
  plt.plot([i for i in range(len(acc))], acc, label="accuracy")
  plt.show()
