import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def initial_plot(train_data, train_label, x_test, y_true):
    plt.scatter(train_data,
                train_label,
                marker='+',
                label='Training data',
                color='black')

    plt.plot(x_test, y_true, label='Truth')

    plt.title('Noisy Training Data and Ground Truth')
    plt.legend()
    plt.show()


def pred_plot(train_data, train_label, x_test, y_true, pred_lst, pred_mean,
              pred_std):
    plt.plot(x_test, pred_mean, c='royalblue', label='Mean Pred')
    plt.fill_between(x_test.reshape(-1, ),
                     pred_mean - 3 * pred_std,
                     pred_mean + 3 * pred_std,
                     color='cornflowerblue',
                     alpha=.5,
                     label='Epistemic Uncertainty (+/- 3 std)')
    # plt.fill_between(x_test.reshape(-1),
    #                  np.percentile(pred_lst, 2.5, axis=0),
    #                  np.percentile(pred_lst, 97.5, axis=0),
    #                  alpha=0.6,
    #                  color='cornflowerblue',
    #                  label='95% Confidence')

    plt.scatter(train_data,
                train_label,
                marker='+',
                color='black',
                label='Training Data')
    plt.plot(x_test, y_true, color='grey', label='Truth')
    plt.legend()
    plt.show()


def plot_hist(net):
    pred_hist = [
        net(Variable(torch.Tensor(np.matrix([0.0])))).data.numpy()[0][0]
        for _ in range(100)
    ]
    plt.hist(pred_hist, density=True, bins=30, label='x=0.0')

    pred_hist = [
        net(Variable(torch.Tensor(np.matrix([0.9])))).data.numpy()[0][0]
        for _ in range(100)
    ]
    # print(pred_hist)
    plt.hist(pred_hist, density=True, bins=30, label='x=0.9')
    plt.xlabel("Sampled y's")
    plt.ylabel("y values")
    plt.legend()
    plt.show()
