import numpy as np
import matplotlib.pyplot as plt
from .bbb_hparams import reg_hparams

from .bbb_model import BayesianNetwork
from .bbb_torch_colab import *
from .bbb_train import run_new

np.random.seed(3)


def Gaussian(x, mu, sigma):
    return 1 / (sigma * (2 * np.pi)**0.5) * np.exp(-0.5 *
                                                   ((x - mu) / sigma)**2)


def MoG(x, s1, s2, pi, noise):
    g1 = Gaussian(x, 0, s1)
    g2 = Gaussian(x, 0, s2)
    return pi * g1 + (1 - pi) * g2


def reg_data(hp):
    train_data = np.linspace(0.0, 0.6, hp.train_size).reshape(-1, 1)
    test_data = np.linspace(0.0, 0.6, hp.test_size).reshape(-1, 1)

    train_target = MoG(train_data, hp.sigma1, hp.sigma2, hp.pi)
    true_y = MoG(train_data, hp.sigma1, hp.sigma2, hp.pi)

    test_target = MoG(test_data, hp.sigma1, hp.sigma2, hp.pi)

    return train_data, train_target, test_data, test_target, true_y


if __name__ == '__main__':
    hp = reg_hparams()
    train_data, train_target, test_data, test_target, true_y = reg_data(hp)

    # hp.num_batch = int(len(train_data) / hp.batch_size)
    # net = BayesianNetwork(1, 1, hp).to(hp.device)
    # losses = run_new(net, train_data, train_target, test_data, test_target, hp)

    # hp.n_train_batches = int(len(train_data) / hp.batch_size)
    # net = BNN(len(train_data[0]), 1, hp.sigma_prior)
    # run(train_data, train_target, test_data, test_target, hp)
