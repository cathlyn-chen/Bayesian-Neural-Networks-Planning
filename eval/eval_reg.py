import torch
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal

from ..utils.plot import *


def eval_reg(net, x_test, n_samples):
    with torch.no_grad():
        pred_lst = [
            net(torch.Tensor(x_test)).data.numpy().squeeze(1)
            for _ in range(n_samples)
        ]

        pred = np.array(pred_lst).T
        pred_mean = pred.mean(axis=1)
        pred_std = pred.std(axis=1)
    return pred_lst, pred_mean, pred_std


def eval_like(net, x_test, y_true, hp):
    _, pred_mean, pred_std = eval_reg(net, x_test, hp.pred_samples)

    y_true = torch.tensor(y_true.reshape(-1, 1))

    N = len(x_test)
    # print(pred_mean[0], pred_mean[1])

    # print(Normal(pred_mean[0], pred_std[0]).log_prob(y_true[0]))

    like = Normal(pred_mean, pred_std).log_prob(y_true)

    print(like.shape)

    return like


def eval_mse(pred, target):
    return (np.square(pred - np.array(target).reshape(-1, 1))).mean()
