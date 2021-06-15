import math
import numpy as np
import torch


class BBBHparams(object):
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 10
        self.n_input = 1
        self.n_output = 1
        self.hidden_units = 32


def mnist_hp():
    hp = BBBHparams()
    hp.batch_size = 1000
    hp.test_batch_size = 5

    hp.classes = 10
    hp.n_epochs = 10
    hp.samples = 2
    hp.test_samples = 10

    hp.pi = 0.5
    hp.sigma1 = torch.FloatTensor([math.exp(-0)])
    hp.sigma2 = torch.FloatTensor([math.exp(-6)])

    hp.task = 'classification'
    hp.activation = 'relu'

    return hp


def reg_hp():
    hp = BBBHparams()
    hp.noise = 0.03
    hp.n_epochs = 3000

    # hp.sigma_prior1 = torch.FloatTensor([math.exp(-0)])
    # hp.sigma_prior2 = torch.FloatTensor([math.exp(-6)])
    hp.sigma_prior1 = 1.5
    hp.sigma_prior2 = 0.1
    hp.pi = 0.5
    hp.noise_tol = .1
    hp.prior = "gaussian"
    hp.n_samples = 3
    hp.learning_rate = 0.01
    hp.task = 'regression'
    hp.activation = 'sigmoid'

    # MoG
    hp.m1 = 1.2
    hp.s1 = 0.3
    hp.m2 = 2.4
    hp.s2 = 0.6

    hp.train_size = 90
    hp.test_size = 30
    hp.n_train_batches = 1
    hp.batch_size = int(hp.train_size / hp.n_train_batches)

    # hp.sigma_prior = float(np.exp(-3))

    # hp.n_samples = 3
    # hp.classes = 1

    return hp


def colab_hparams():
    hp = BBBHparams()
    hp.n_samples = 3
    hp.sigma_prior = float(np.exp(-3))
    hp.learning_rate = 0.001
    hp.batch_size = 1000

    return hp