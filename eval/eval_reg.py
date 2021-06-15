import torch
import numpy as np
from torch.autograd import Variable


def eval_reg(x_test, net):
    with torch.no_grad():
        pred_lst = [
            net(Variable(torch.Tensor(x_test))).data.numpy().squeeze(1)
            for _ in range(100)
        ]

        pred = np.array(pred_lst).T
        pred_mean = pred.mean(axis=1)
        pred_std = pred.std(axis=1)
    return pred_lst, pred_mean, pred_std

