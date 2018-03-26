#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:35:58 2017

@author: owen
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision

class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('data/data.npy')
        y = np.load('data/label.npy')  
        
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).unsqueeze(1).float()
        self.y_data = torch.from_numpy(y).unsqueeze(1).float()

    def __len__(self):
        
        return self.len

    def __getitem__(self, idx):
        
        return self.x_data[idx], self.y_data[idx]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 0)
        self.conv2 = nn.Conv2d(16, 8, 7, 1, 0)
        self.fc1 = nn.Linear(4*4*8, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fc1(out))
        
        return out

# Specify the newtork architecture
net = Net()

# Specify the training dataset
dataset = Dataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=128,
                          shuffle=True)


# Visualize the dataset
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.title('Visualize the dataset')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))



# Specify the loss function
criterion = nn.BCELoss()

# Specify the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.99, weight_decay=5e-4)

max_epochs = 1000

loss_np = np.zeros((max_epochs))
accuracy = np.zeros((max_epochs))

for epoch in range(max_epochs):
    for i, data in enumerate(train_loader, 0):
        
        # Get inputs and labels from data loader 
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        # Feed the input data into the network 
        y_pred = net(inputs)
        
        # Calculate the loss using predicted labels and ground truth labels
        loss = criterion(y_pred, labels)
        
        print("epoch: ", epoch, "loss: ", loss.data[0])
        
        # zero gradient
        optimizer.zero_grad()
        
        # backpropogates to compute gradient
        loss.backward()
        
        # updates the weghts
        optimizer.step()
        
        # convert predicted laels into numpy
        y_pred_np = y_pred.data.numpy()
        
        # calculate the training accuracy of the current model
        pred_np = np.where(y_pred_np>0.5, 1, 0) 
        label_np = labels.data.numpy().reshape(len(labels),1)
        
        correct = 0
        for j in range(y_pred_np.shape[0]):
            if pred_np[j,:] == label_np[j,:]:
                correct += 1
        
        accuracy[epoch] = float(correct)/float(len(label_np))
        
        loss_np[epoch] = loss.data.numpy()


print("final training accuracy: ", accuracy[max_epochs-1])

epoch_number = np.arange(0,max_epochs,1)

# Plot the loss over epoch
plt.figure()
plt.plot(epoch_number, loss_np)
plt.title('loss over epoches')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss')

# Plot the training accuracy over epoch
plt.figure()
plt.plot(epoch_number, accuracy)
plt.title('training accuracy over epoches')
plt.xlabel('Number of Epoch')
plt.ylabel('accuracy')