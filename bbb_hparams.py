import math
import torch


class Hparams:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


def mnist():
    hparams = Hparams()
    hparams.batch_size = 1000
    hparams.test_batch_size = 5

    hparams.classes = 10
    hparams.trian_epochs = 20
    hparams.samples = 2
    hparams.test_samples = 10

    hparams.pi = 0.5
    hparams.sigma_1 = torch.FloatTensor([math.exp(-0)])
    hparams.sigma_2 = torch.FloatTensor([math.exp(-6)])

    return hparams