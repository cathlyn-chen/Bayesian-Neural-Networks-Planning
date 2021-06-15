import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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
    def __init__(self, hp):
        super(BNN, self).__init__()
        self.l1 = BNNLayer(hp.n_input, hp.hidden_units, hp.sigma_prior1)
        self.l1_relu = nn.ReLU()
        self.l2 = BNNLayer(hp.hidden_units, hp.hidden_units, hp.sigma_prior1)
        self.l2_relu = nn.ReLU()
        self.l3 = BNNLayer(hp.hidden_units, hp.n_output, hp.sigma_prior1)
        self.l3_softmax = nn.Softmax()
        self.hp = hp

    def forward(self, X, infer=False):
        output = self.l1_relu(self.l1(X, infer))
        output = self.l2_relu(self.l2(output, infer))
        if self.hp.task == 'classification':
            output = self.l3_softmax(self.l3(output, infer))
        elif self.hp.task == 'regression':
            output = self.l3(output, infer)
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw

    def sample_elbo(self, X, y):
        s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
        for _ in range(self.hp.n_samples):
            output = self(X)
            sample_log_pw, sample_log_qw = self.get_lpw_lqw()
            sample_log_likelihood = log_gaussian(y, output,
                                                 self.hp.sigma_prior1).sum()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood

        log_pw, log_qw, log_likelihood = s_log_pw / self.hp.n_samples, s_log_qw / self.hp.n_samples, s_log_likelihood / self.hp.n_samples

        loss = ((1. / self.hp.n_train_batches) * (log_qw - log_pw) -
                log_likelihood).sum() / float(self.hp.batch_size)

        return loss