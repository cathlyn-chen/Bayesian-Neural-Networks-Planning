import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

from .hparams import colab_hparams


def get_mnist():
    train = datasets.MNIST(root="./data",
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

    train_data = [np.array(torch.flatten(data[:-1][0])) for data in train]
    train_label = [data[-1] for data in train]

    test = datasets.MNIST(root="./data",
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

    test_data = [np.array(torch.flatten(data[:-1][0])) for data in test]
    test_label = [data[-1] for data in test]

    train_label_one_hot = np.zeros((len(train_label), 10))
    train_label_one_hot[np.arange(len(train_label)), train_label] = 1
    train_label_one_hot = torch.tensor(train_label_one_hot,
                                       dtype=float).float()

    test_label_one_hot = np.zeros((len(test_label), 10))
    test_label_one_hot[np.arange(len(test_label)), test_label] = 1
    test_label_one_hot = torch.tensor(test_label_one_hot).float()

    return train_data, train_label_one_hot, test_data, test_label_one_hot


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) -
                 np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_rho(x, mu, rho):
    return float(-0.5 * np.log(2 * np.pi)) - torch.log(
        torch.log(1 + torch.exp(rho))) - (x - mu)**2 / (
            2 * torch.log(1 + torch.exp(rho))**2)


class BNNLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super(BNNLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        self.W_mu = nn.Parameter(
            torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.W_rho = nn.Parameter(
            torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.lpw = 0
        self.lqw = 0

    def sample_epsilon(self):
        return Variable(
            torch.Tensor(self.n_input, self.n_output).normal_(
                0, self.sigma_prior)), Variable(
                    torch.Tensor(self.n_output).normal_(0, self.sigma_prior))

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(
                X.size()[0], self.n_output)
            return output

        # Step 1
        epsilon_W, epsilon_b = self.sample_epsilon()

        # Step 2
        W = self.W_mu + torch.log(1 + torch.exp(self.W_rho)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * epsilon_b

        # Step 3
        self.lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(
            b, 0, self.sigma_prior).sum()

        # Step 4
        self.lqw = log_gaussian_rho(W, self.W_mu,
                                    self.W_rho).sum() + log_gaussian_rho(
                                        b, self.b_mu, self.b_rho).sum()

        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        return output


class BNN(nn.Module):
    def __init__(self, n_input, n_output, hidden_units, sigma_prior):
        super(BNN, self).__init__()
        self.l1 = BNNLayer(n_input, hidden_units, sigma_prior)
        self.l1_relu = nn.ReLU()
        self.l2 = BNNLayer(hidden_units, hidden_units, sigma_prior)
        self.l2_relu = nn.ReLU()
        self.l3 = BNNLayer(200, n_output, sigma_prior)
        self.l3_softmax = nn.Softmax()

    def forward(self, X, infer=False):
        output = self.l1_relu(self.l1(X, infer))
        output = self.l2_relu(self.l2(output, infer))
        output = self.l3_softmax(self.l3(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw


def forward_pass_samples(net, X, y, hp):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(hp.n_samples):
        output = net(X)
        sample_log_pw, sample_log_qw = net.get_lpw_lqw()
        sample_log_likelihood = log_gaussian(y, output, hp.sigma_prior).sum()
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    return s_log_pw / hp.n_samples, s_log_qw / hp.n_samples, s_log_likelihood / hp.n_samples


def loss_fn(l_pw, l_qw, l_likelihood, hp):
    return ((1. / hp.n_train_batches) *
            (l_qw - l_pw) - l_likelihood).sum() / float(hp.batch_size)


def run(train_data, train_label, test_data, test_label, hp):
    n_input = len(train_data[0])

    # Initialize network
    net = BNN(n_input, 10, hp.sigma_prior)
    optimizer = torch.optim.Adam(net.parameters(), lr=hp.learning_rate)

    log_pw, log_qw, log_likelihood = 0., 0., 0.
    test_acc = []
    test_err = []

    # Training loop
    for e in range(hp.n_epochs):
        errs = []

        # Batch training
        for b in range(hp.n_train_batches):
            net.zero_grad()

            # Obtain minibatch
            X = Variable(
                torch.Tensor(train_data[b * hp.batch_size:(b + 1) *
                                        hp.batch_size]))
            y = Variable(
                torch.Tensor(train_label[b * hp.batch_size:(b + 1) *
                                         hp.batch_size]))

            log_pw, log_qw, log_likelihood = forward_pass_samples(
                net, X, y, hp)
            loss = loss_fn(log_pw, log_qw, log_likelihood, hp)

            errs.append(loss.data.numpy())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            X_test = Variable(torch.Tensor(test_data))

            # Predict with softmax
            pred_test = net(X_test, infer=True)
            _, out_test = torch.max(pred_test, 1)

            out_test = out_test.data.numpy()
            print(out_test.shape, out_test[:3])

            test_label = np.array(test_label)

            # Evaluate
            acc_test = [
                out_test[i] == test_label[i] for i in range(out_test.shape[0])
            ]
            print(len(acc_test), sum(acc_test), out_test.shape[0])

            acc_test = np.count_nonzero(acc_test) / out_test.shape[0]

            err_test = 1 - acc_test

        test_acc.append(acc_test)
        test_err.append(err_test)

        print('epoch', e, 'loss', np.mean(errs), 'test acc', acc_test)

    return test_acc, test_err


def plot(err):
    # plt.plot(test_acc, label="Test Accuracy")
    plt.plot(err)
    plt.xlabel('Epoch')
    plt.ylabel('Test Error (%)')
    plt.ylim(0, 9)
    plt.show()


if __name__ == '__main__':
    hp = colab_hparams()

    train_data, train_label, test_data, test_label = get_mnist()

    hp.n_train_batches = int(len(train_data) / hp.batch_size)

    test_acc, test_err = run(train_data, train_label, test_data, test_label,
                             hp)

    test_err_plt = [err * 100 for err in test_err]

    plot(test_err_plt)

    print(f'Test error is {test_err_plt[-1]:.2f}% after {hp.n_epochs} epochs')
