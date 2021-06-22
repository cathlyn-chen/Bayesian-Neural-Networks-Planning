import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..hparams import reg_hp
from ..utils.plot import *

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


def multi_normal(x, hp):
    x_m = x - hp.mean
    return (1. / (np.sqrt((2 * np.pi)**2 * np.linalg.det(hp.covariance))) *
            np.exp(-(np.linalg.solve(hp.covariance, x_m).T.dot(x_m)) / 2))


def poly_2d(x, sigma):
    # e1 = np.random.randn() * sigma
    # e2 = np.random.randn() * sigma
    x1, x2 = x
    # return -(x1 + e1)**2 - (x1 + e1) + (x2 + e2)**2
    return -x1**2 - x1 + x2**2


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


def ncp_data(hp):
    rng = np.random.RandomState(123)

    def f(x):
        """Sinusoidal function."""
        return 0.5 * np.sin(25 * x) + 0.5 * x

    def noise(x, slope, rng=np.random):
        """Create heteroskedastic noise."""
        noise_std = np.maximum(0.0, x + 1.0) * slope
        return rng.normal(0, noise_std).astype(np.float32)

    def select_bands(x, y, mask):
        assert x.shape[0] == y.shape[0]

        num_bands = len(mask)

        if x.shape[0] % num_bands != 0:
            raise ValueError(
                'size of first dimension must be a multiple of mask length')

        data_mask = np.repeat(mask, x.shape[0] // num_bands)
        return [arr[data_mask] for arr in (x, y)]

    def select_subset(x, y, num, rng=np.random):
        assert x.shape[0] == y.shape[0]

        choices = rng.choice(range(x.shape[0]), num, replace=False)
        return [x[choices] for x in (x, y)]

    x = np.linspace(-1.0, 1.0, 1000, dtype=np.float32).reshape(-1, 1)

    # Noisy samples from f (with heteroskedastic noise)
    y = f(x) + noise(x, slope=0.2, rng=rng)

    # Select data from 2 of 5 bands (regions)
    x_bands, y_bands = select_bands(x,
                                    y,
                                    mask=[False, True, False, True, False])

    # Select 40 random samples from these regions
    x_train, y_train = select_subset(x_bands,
                                     y_bands,
                                     num=hp.train_size,
                                     rng=rng)

    return x_train, y_train, x, f(x)


def gaussian_data_2d(hp):
    x_train = multivariate_normal.rvs(hp.mean,
                                      hp.covariance,
                                      size=hp.train_size)

    y_train = np.array([multi_normal(x, hp) for x in x_train])

    # x_test = np.mgrid[-3:3.6:0.3, -2.4:3.6:0.3].reshape(2, -1).T

    x_test = hp.grid.reshape(2, -1).T

    y_true = np.array([multi_normal(x, hp) for x in x_test])

    # y_true = multivariate_normal(hp.mean, hp.covariance)

    # x1, x2 = hp.grid
    # pos = np.empty(x1.shape + (2, ))
    # pos[:, :, 0] = x1
    # pos[:, :, 1] = x2

    # y_true = y_true.pdf(pos)

    return x_train, y_train, x_test, y_true


def poly_data_2d(hp):

    data = hp.grid.reshape(2, -1).T
    np.random.shuffle(data)
    # print(data.shape)

    x_train = data[:hp.train_size]

    y_train = np.array([poly_2d(x, hp.noise) for x in x_train])

    x_test = hp.grid.reshape(2, -1).T
    y_true = np.array([poly_2d(x, 0) for x in x_test])
    # print(y_true.shape)

    return x_train, y_train, x_test, y_true


def transform_data(data):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaler, scaled_data


def inverse_data(scaler, data):
    ori_data = scaler.inverse_transform(data)
    return ori_data
