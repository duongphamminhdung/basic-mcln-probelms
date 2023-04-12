import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
from datetime import datetime

torch.seed()
# Define the model
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        
        #drop out
        self.drop_out = nn.Dropout(0.2)
        #CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size = (3, 3), stride=1, padding=1)
        #in 1x28x28 out 3x28x28
        
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size= (3, 3), stride=1, padding=1)
        self.act = nn.ReLU()
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
        x = (self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        # import ipdb;ipdb.set_trace()
        x = self.drop_out(self.flat(x))
        x = self.linear1(x)
        x = self.linear2(self.act(x))
        x = self.softmax(x)
        
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on", device)
torch.manual_seed(256)

batch = 256
# Store separate training and validations splits in ./data
training_set = torchvision.datasets.MNIST('./data',download=True,train=True,transform=transforms.ToTensor())
validation_set = torchvision.datasets.MNIST('./data',download=True,train=False,transform=transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(training_set,batch_size=batch,shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set,batch_size=1,shuffle=True)

# Define the model and loss function
model = MnistModel().to(device)
criterion = nn.MSELoss()
timestamp = datetime.now().strftime('%Y%m%d')
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, nesterov=True) 
# optimizer = optim.Adagrad(model.parameters(), lr=0.025)


path = '/content/weights/20230411_SGD'
if os.path.exists(path) and path.split('_')[-1] == optimizer.__class__.__name__:
  state_dict = torch.load(path)
  model.load_state_dict(state_dict)
  print('Loaded model')
# elif path.split('_')[-2] != timestamp:
#   print("Diffirent date, continue?")
elif path.split('_')[-1] != optimizer.__class__.__name__:
  print('Diffirent optimizer')
else:
  print('Model not found')
print('loss_func:', criterion, "optimizer:", optimizer)


NUM_EPOCHS = 200
acc = []
losses = []
best_acc = 90
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
          best_acc = acc[-1]
          model_path = '{}_{}'.format(timestamp, optimizer.__class__.__name__ )
          weight_folder = 'weights'
          os.makedirs(weight_folder, exist_ok=True)
          best_weight_path = os.path.join(weight_folder, model_path)
          torch.save(model.state_dict(), best_weight_path)
          print("Saved at location", best_weight_path)

      # Print progress
      print('''Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f},
Test Loss: {:.4f}, Test Accuracy: {:.2f}'''.format(
          epoch+1, NUM_EPOCHS, train_loss, train_accuracy*100, test_loss, test_accuracy))
      if acc[-1] >= 99 or epoch > 500:
        break
  plt.grid()
  plt.plot([i for i in range(len(acc))], acc, label="accuracy")
  plt.show()
except KeyboardInterrupt:
  plt.grid()
  plt.plot([i for i in range(len(acc))], acc, label="accuracy")
  plt.show()
  print('current best accuracy:', best_acc)
