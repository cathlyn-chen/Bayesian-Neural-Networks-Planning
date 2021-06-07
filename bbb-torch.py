import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {
    'num_workers': 1,
    'pin_memory': True
} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())
