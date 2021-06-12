import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from .bbb_hparams import mnist_hp
from .models.bnn_old import BayesianNetwork
from .bbb_run import run, run_new


def load_mnist(hp):
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '/data', train=True, download=True, transform=transforms.ToTensor()),
                                               batch_size=hp.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '/data', train=False, download=True, transform=transforms.ToTensor()),
                                              shuffle=False)

    return train_loader, test_loader


def get_mnist():
    train = datasets.MNIST(root="./data",
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

    train_data = [np.array(torch.flatten(data[:-1][0])) for data in train]
    train_label = [data[-1] for data in train]

    test = datasets.MNIST(root="./data",
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

    test_data = [np.array(torch.flatten(data[:-1][0])) for data in test]
    test_label = [data[-1] for data in test]

    train_label_one_hot = np.zeros((len(train_label), 10))
    train_label_one_hot[np.arange(len(train_label)), train_label] = 1
    train_label_one_hot = torch.tensor(train_label_one_hot,
                                       dtype=float).float()

    test_label_one_hot = np.zeros((len(test_label), 10))
    test_label_one_hot[np.arange(len(test_label)), test_label] = 1
    test_label_one_hot = torch.tensor(test_label_one_hot).float()

    return train_data, train_label_one_hot, test_data, test_label_one_hot


if __name__ == '__main__':
    hp = mnist_hp()
    # train_loader, test_loader = load_mnist(hp)

    # hp.train_size = len(train_loader.dataset)
    # hp.test_size = len(test_loader.dataset)
    # hp.num_batch = len(train_loader)
    # hp.test_batch = len(test_loader)

    # net = BayesianNetwork(784, 10, 30, hp).to(hp.device)
    # run(net, train_loader, test_loader, hp)

    # torch.save(net.state_dict(), '/model')

    train_data, train_label_one_hot, test_data, test_label_one_hot = get_mnist(
    )

    # run_new(net, train_data, train_label_one_hot, test_data,
    #         test_label_one_hot, hp)
