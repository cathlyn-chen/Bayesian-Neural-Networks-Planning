import torch
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal

from ..utils.plot import *


def eval_reg(net, x, n):
    with torch.no_grad():
        pred_lst = [
            net(torch.Tensor(x)).data.numpy().squeeze(1) for _ in range(n)
        ]  #(n, x_test.shape)

        pred = np.array(pred_lst).T

        # print(np.array(pred_lst).shape, pred.shape)

        pred_mean = pred.mean(axis=1)
        pred_std = pred.std(axis=1)

        # print(pred_mean.shape, np.mean(pred_std))
    return pred_lst, pred_mean, pred_std


def eval_like(net, x, y, hp):
    with torch.no_grad():
        _, pred_mean, pred_std = eval_reg(net, x, hp.eval_samples)

        y = torch.tensor(y.reshape(-1, 1))
        # print(y.shape)
        # print(pred_mean.shape, y.shape)

        # print(pred_mean[0], pred_mean[1])
        # print(Normal(pred_mean[0], pred_std[0]).log_prob(y[0]))

        like = np.array([
            Normal(pred_mean[i], pred_std[i]).log_prob(y[i])
            for i in range(len(y))
        ])

        # print(like.shape)

        # plot_like(x, like)

    return like


def eval_mse(pred, target):
    return (np.square(pred - np.array(target).reshape(-1, 1))).mean()
