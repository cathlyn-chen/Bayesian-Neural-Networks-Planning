import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from ..hparams import reg_hp
from ..utils.plot import *

# from ..models.bnn1 import BNN
# from ..models.bnn2 import BNN
from ..models.bnn3 import BNN

np.random.seed(3)
''' Ground truth functions for regression '''


def toy_reg(X, noise):
    epsilon = np.random.normal(0, noise, size=X.shape)
    return X**3 - X + epsilon


def paper_reg(X, sigma):
    epsilon = np.random.normal(0, sigma, size=X.shape)
    return X + 0.3 * np.sin(2 * np.pi * (X + epsilon)) + 0.3 * np.sin(
        4 * np.pi * (X + epsilon)) + epsilon


def Gaussian(X, mu, sigma):
    return 1 / (sigma * (2 * np.pi)**0.5) * np.exp(-0.5 *
                                                   ((X - mu) / sigma)**2)


def MoG(X, m1, s1, m2, s2, pi, noise):
    g1 = Gaussian(X, m1, s1)
    g2 = Gaussian(X, m2, s2)

    epsilon = np.random.normal(0, noise, size=X.shape)

    return pi * (g1 + epsilon) + (1 - pi) * (g2 + epsilon)


def f(X, sigma):
    epsilon = np.random.randn(*X.shape) * sigma
    return 10 * np.sin(2 * np.pi * (X)) + epsilon


def toy_function(X):
    return -X**4 + 3 * X**2 + 1


def poly(X, sigma):
    epsilon = np.random.randn(*X.shape) * sigma
    return 2 * (X + epsilon)**2 + 2


''' Data for regression from ground truth '''


def poly_data(hp):
    train_neg = np.linspace(-4.0, -1.0, 30).reshape(-1, 1)
    train_pos = np.linspace(1.0, 4.0, 30).reshape(-1, 1)
    train_data = np.concatenate((train_neg, train_pos), axis=0)
    train_label = poly(train_data, hp.noise)

    x_test = np.linspace(-6.0, 6.0, 1000).reshape(-1, 1)
    y_true = poly(x_test, 0.0)

    return train_data, train_label, x_test, y_true


def f_data(hp):
    train_data = np.linspace(-0.5, 0.5, 32).reshape(-1, 1)
    train_label = f(train_data, sigma=hp.noise)
    x_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
    y_true = f(x_test, sigma=0.0)
    return train_data, train_label, x_test, y_true


def MoG_data(hp):
    # Different data densities
    train_low = np.linspace(0.0, 0.3, 30).reshape(-1, 1)
    train_high = np.linspace(0.9, 1.5, 60).reshape(-1, 1)
    train_medium = np.linspace(2.4, 3.3, 30).reshape(-1, 1)

    train_data = np.concatenate((train_low, train_high, train_medium), axis=0)

    train_label = MoG(train_data, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, hp.noise)

    x_test = np.linspace(-0.3, 4.2, 1000).reshape(-1, 1)
    y_true = MoG(x_test, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, 0)

    return train_data, train_label, x_test, y_true


def MoG_data_val(hp):
    data = np.linspace(0.3, 3.3, hp.train_size + hp.val_size).reshape(-1, 1)

    np.random.shuffle(data)

    x_train = data[:hp.train_size]
    x_val = data[hp.train_size:]

    y_train = MoG(x_train, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, hp.noise)
    y_val = MoG(x_val, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, hp.noise)

    x_test = np.linspace(-0.15, 4.2, 1000).reshape(-1, 1)
    y_true = MoG(x_test, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, 0)

    return x_train, y_train, x_val, y_val, x_test, y_true


def toy_reg_data(hp):
    area_size = int(hp.train_size / 3)
    train_low = np.linspace(0.0, 0.15, area_size).reshape(-1, 1)
    train_high = np.linspace(0.3, 0.6, area_size).reshape(-1, 1)
    train_medium = np.linspace(0.9, 1.2, area_size).reshape(-1, 1)

    train_data = np.concatenate((train_low, train_high, train_medium), axis=0)

    train_label = toy_reg(train_data, hp.noise)

    x_test = np.linspace(-0.3, 1.8, 1000).reshape(-1, 1)
    y_true = toy_reg(x_test, 0)

    return train_data, train_label, x_test, y_true


def paper_reg_data(hp):
    train_low = np.linspace(0.0, 0.15, 15).reshape(-1, 1)
    train_high = np.linspace(0.15, 0.3, 30).reshape(-1, 1)
    train_medium = np.linspace(0.3, 0.45, 15).reshape(-1, 1)

    train_data = np.concatenate((train_low, train_high, train_medium), axis=0)

    train_label = paper_reg(train_data, hp.noise)

    x_test = np.linspace(-0.3, 1.2, 1000).reshape(-1, 1)
    y_true = paper_reg(x_test, 0)

    return train_data, train_label, x_test, y_true
