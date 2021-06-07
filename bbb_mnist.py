import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms




train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    '/data', train=True, download=True, transform=transforms.ToTensor()),
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(
    '/data', train=False, download=True, transform=transforms.ToTensor()),
                                          batch_size=TEST_BATCH_SIZE,
                                          shuffle=False)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0
