import math
import numpy as np
import torch


class BBBHparams(object):
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 10


def mnist_hp():
    hparams = BBBHparams()
    hparams.batch_size = 1000
    hparams.test_batch_size = 5

    hparams.classes = 10
    hparams.n_epochs = 10
    hparams.samples = 2
    hparams.test_samples = 10

    hparams.pi = 0.5
    hparams.sigma1 = torch.FloatTensor([math.exp(-0)])
    hparams.sigma2 = torch.FloatTensor([math.exp(-6)])

    return hparams


def reg_hparams():
    hp = BBBHparams()
    hp.train_size = 90
    hp.test_size = 30
    hp.noise = 0.02

    hp.sigma1 = 0.1
    hp.sigma2 = 0.6
    hp.pi = 0.3

    hp.batch_size = 30

    hp.sigma_prior = float(np.exp(-3))
    hp.learning_rate = 0.001
    hp.n_samples = 3

    hp.samples = 3
    hp.classes = 1

    return hp


def colab_hparams():
    hp = BBBHparams()
    hp.n_samples = 3
    hp.sigma_prior = float(np.exp(-3))
    hp.learning_rate = 0.001
    hp.batch_size = 1000

    return hp