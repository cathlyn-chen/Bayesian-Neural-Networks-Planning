import math
import torch


def hparams():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOADER_KWARGS = {
        'num_workers': 1,
        'pin_memory': True
    } if torch.cuda.is_available() else {}
    print(torch.cuda.is_available())

    BATCH_SIZE = 1000
    TEST_BATCH_SIZE = 5

    CLASSES = 10
    TRAIN_EPOCHS = 20
    SAMPLES = 2
    TEST_SAMPLES = 10

    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])