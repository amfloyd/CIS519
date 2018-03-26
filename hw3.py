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
    def __init__(self):
        X = np.load('hw_data/train_images.npy')
        y = np.load('hw_data/train_labels.npy')

        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).unsqueeze(1).float()
        self.y_data = torch.from_numpy(y).unsqueeze(1).float()

