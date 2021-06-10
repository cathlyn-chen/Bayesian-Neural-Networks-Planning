import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable

from .bbb_hparams import reg_hparams
# from .bbb_model import BayesianNetwork
from .bbb_model2 import BNN
# from .bbb_torch_colab import *
# from .bbb_run import run_new

np.random.seed(3)

# Ground Truth Functions


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


def MoG_data(hp):
    # Different data densities
    train_low = np.linspace(0.0, 0.15, 60).reshape(-1, 1)
    train_high = np.linspace(0.3, 0.6, 15).reshape(-1, 1)
    train_medium = np.linspace(0.9, 1.2, 6).reshape(-1, 1)

    train_data = np.concatenate((train_low, train_high, train_medium), axis=0)

    train_label = MoG(train_data, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, hp.noise)

    x_plot = np.linspace(-0.3, 1.5, 1000).reshape(-1, 1)
    true_y = MoG(x_plot, hp.m1, hp.s1, hp.m2, hp.s2, hp.pi, 0)

    return train_data, train_label, x_plot, true_y


def toy_reg_data(hp):
    train_low = np.linspace(0.0, 0.15, 15).reshape(-1, 1)
    train_high = np.linspace(0.3, 0.6, 30).reshape(-1, 1)
    train_medium = np.linspace(0.9, 1.2, 60).reshape(-1, 1)

    train_data = np.concatenate((train_low, train_high, train_medium), axis=0)

    train_label = toy_reg(train_data, hp.noise)

    x_plot = np.linspace(-0.3, 1.5, 1000).reshape(-1, 1)
    true_y = toy_reg(x_plot, 0)

    return train_data, train_label, x_plot, true_y


def initial_plot(train_data, train_label, x_plot, true_y):
    plt.scatter(train_data,
                train_label,
                marker='+',
                label='Training data',
                color='black')

    plt.plot(x_plot, true_y, label='Truth')

    plt.title('Noisy Training Data and Ground Truth')
    plt.legend()
    plt.show()


def train_bbb(train_data, train_label, hp):
    # net = BayesianNetwork(1, 1, hp)
    net = BNN(32, prior_var=10)
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    for e in range(hp.n_epochs):
        net.zero_grad()
        X = torch.tensor([float(data) for data in train_data]).reshape(-1, 1)
        y = torch.tensor([float(label) for label in train_label])
        loss = net.sample_elbo(X, y, 1)
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print('epoch: {}'.format(e + 1), 'loss', loss.item())
    print('Finished Training')

    return net


def eval(x_plot, net):
    with torch.no_grad():

        pred_lst = [
            net(Variable(torch.Tensor(x_plot))).data.numpy().squeeze(1)
            for _ in range(100)
        ]

        pred = np.array(pred_lst).T
        pred_mean = pred.mean(axis=1)
        pred_std = pred.std(axis=1)
    return pred_mean, pred_std


def pred_plot(train_data, train_label, x_plot, true_y, pred_mean, pred_std):
    plt.plot(x_plot, pred_mean, c='royalblue', label='Mean Pred')
    plt.fill_between(x_plot.reshape(-1, ),
                     pred_mean - 2 * pred_std,
                     pred_mean + 2 * pred_std,
                     color='cornflowerblue',
                     alpha=.5,
                     label='Epistemic Uncertainty (+/- 2 std)')
    # print(test_data.reshape(30,))

    plt.scatter(train_data,
                train_label,
                marker='+',
                color='black',
                label='Training Data')
    # plt.scatter(test_data, test_pred, marker='+', label='Test Pred', color='black')
    plt.plot(x_plot, true_y, color='grey', label='Truth')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    hp = reg_hparams()
    train_data, train_label, x_plot, true_y = toy_reg_data(hp)

    print(train_data.shape, train_label.shape)

    # initial_plot(train_data, train_label, x_plot, true_y)
    net = train_bbb(train_data, train_label, hp)

    pred_mean, pred_std = eval(x_plot, net)

    pred_plot(train_data, train_label, x_plot, true_y, pred_mean, pred_std)

    # hp.num_batch = int(len(train_data) / hp.batch_size)
    # net = BayesianNetwork(1, 1, hp).to(hp.device)
    # losses = run_new(net, train_data, train_target, test_data, test_target, hp)

    # hp.n_train_batches = int(len(train_data) / hp.batch_size)
    # net = BNN(len(train_data[0]), 1, hp.sigma_prior)
    # run(train_data, train_target, test_data, test_target, hp)
