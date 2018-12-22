import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, transforms
from torchvision import utils as tvutils

import os

from sklearn.metrics import accuracy_score

from tensorboardX import SummaryWriter

import numpy as np

class Net(nn.Module):
    def __init__(self, size, input_depth):
        super().__init__()

        self.size = size
        self.input_depth = input_depth
        self.conv_depth = 64
        self.hidden_dim = 1024

        # Add 3 conv layers
        self.conv_layers = nn.Sequential(
                nn.Conv2d(self.input_depth,
                              self.conv_depth,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(self.conv_depth,
                              self.conv_depth * 2,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(self.conv_depth * 2,
                              self.conv_depth * 4,
                              kernel_size=3,
                              stride=1,
                              padding=1),
                nn.ReLU(),
                nn.Dropout(0.5)
            )


        self.cur_size = self.size // 4
        self.cur_depth = self.conv_depth * 4

        # Add 2 linear layers
        self.linear_layers = nn.Sequential(
                nn.Linear(self.cur_size * self.cur_size * self.cur_depth, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = F.relu(x)

        # Move to GPU 1 since the linear layers are on GPU 1
        x = x.view(-1, self.cur_size * self.cur_size * self.cur_depth)

        x = self.linear_layers(x)

        return x

datadir = './data'
if os.path.exists(datadir) == False:
    os.mkdir(datadir)

size = 28
input_depth = 1
batch_size = 1024

# Training dataset
training_dataset = datasets.MNIST(
            root=datadir,
            train=True,
            download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
            ]))

training_set_size = int(0.8 * len(training_dataset))
validation_set_size = len(training_dataset) - training_set_size

training_dataset, valid_dataset = data.random_split(
        training_dataset, 
        [training_set_size, validation_set_size])

trainloader = data.DataLoader(
                training_dataset,
                batch_size=batch_size,
                shuffle=True)

# Validation dataset
validloader = data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=True)

# Test dataset
test_dataset = datasets.MNIST(
            root=datadir,
            train=True,
            download=True,
            transform = transforms.Compose([
                transforms.ToTensor(),
            ]))
testloader = data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True)

net = Net(size, input_depth)
net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

writer = SummaryWriter('runs/exp_{}'.format(net.conv_depth))

num_epochs = 10
for i in range(num_epochs):
    net.train()
    train_loss = 0.0
    num_batches = 0
    for j, (inputs, labels) in enumerate(trainloader):

        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        preds = net(inputs)
        loss = loss_fn(preds, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.data.item()
        num_batches += 1

    writer.add_scalar('loss/train', train_loss / num_batches, i + 1)

    print("Epoch {}: avg training loss = {}\n".format(i + 1, train_loss / num_batches))
    writer.add_text("Loss", "Epoch {}: avg training loss = {}\n".format(i + 1, train_loss / num_batches), i + 1)

    valid_loss = 0.0
    num_batches = 0
    net.eval()
    for inputs, labels in validloader:
        inputs, labels = inputs.cuda(), labels.cuda()

        preds = net(inputs)
        loss = loss_fn(preds, labels)

        valid_loss += loss.data.item()
        num_batches += 1

    writer.add_scalar('loss/validation', train_loss / num_batches, i + 1)
        
net.eval()
ground_truth = []
predictions = []

incorrect_predictions = {}
for inputs, labels in testloader:

    inputs, labels = inputs.cuda(), labels.cuda()
    
    _, preds = F.softmax(net(inputs), dim=1).max(1)

    ground_truth.extend(labels.cpu().numpy())
    predictions.extend(preds.cpu().numpy())

    for i in range(len(labels)):
        if preds[i] != labels[i]:
            if preds[i].item() not in incorrect_predictions.keys():
                incorrect_predictions[preds[i].item()] = []

            incorrect_predictions[preds[i].item()].append(inputs[i].cpu().numpy())

print("Accuracy: {0:.2f}".format(100 * accuracy_score(ground_truth, predictions)))
writer.add_text('accuracy', "Accuracy: {0:.2f}".format(100 * accuracy_score(ground_truth, predictions)))

for i in incorrect_predictions.keys():
    img_tensor = torch.from_numpy(np.array(incorrect_predictions[i]))
    img_grid = tvutils.make_grid(img_tensor)
    writer.add_image('incorrect/{}'.format(i), img_grid, 0)
