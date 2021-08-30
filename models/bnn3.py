import numpy as np
import torch
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.modules.loss import NLLLoss


class ScaleMixtureGaussian(object):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.gaussian1 = torch.distributions.Normal(0, hp.sigma_prior1)
        self.gaussian2 = torch.distributions.Normal(0, hp.sigma_prior2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.hp.pi * prob1 + (1 - self.hp.pi) * prob2)).sum()


class BNNLayer(nn.Module):
    def __init__(self, n_input, n_output, hp):
        # Initialize BNN layer
        super().__init__()
        self.hp = hp

        # Initialize mu and rho parameters for layer's weights
        self.w_mu = nn.Parameter(torch.zeros(n_output, n_input))
        self.w_rho = nn.Parameter(torch.zeros(n_output, n_input))

        # Initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(n_output))
        self.b_rho = nn.Parameter(torch.zeros(n_output))

        # Initialize prior distribution for all of the weights and biases
        if hp.prior == 'gaussian':
            self.prior = torch.distributions.Normal(0, hp.sigma_prior1)
        elif hp.prior == 'scale_mixture':
            self.prior = ScaleMixtureGaussian(hp)
        elif hp.prior == 'ncp':
            self.prior = hp.data_prior

        # Initialize weight samples - calculated whenever the layer makes a prediction
        self.w = None
        self.b = None

    def forward(self, input):
        # Sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # Sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # Log prior - evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # Log variational posterior - evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data,
                             torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data,
                             torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(
            self.w).sum() + self.b_post.log_prob(self.b).sum()

        forward = F.linear(input, self.w, self.b)
        return forward


class BNN(nn.Module):
    def __init__(self, hp):
        # Initialize the network using the BBB layer
        super().__init__()
        self.hp = hp

        # Layers
        self.input_layer = BNNLayer(hp.n_input, hp.hidden_units, hp)
        self.hidden_layer = BNNLayer(hp.hidden_units, hp.hidden_units, hp)
        self.output_layer = BNNLayer(hp.hidden_units, hp.n_output, hp)

        # Activation function
        if hp.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif hp.activation == 'relu':
            self.act = nn.ReLU()
        elif hp.activation == 'tanh':
            self.act = nn.Tanh()

        if hp.task == 'classification':
            self.softmax = nn.Softmax()

        # Noise tolerance used to calculate likelihood
        self.noise_tol = hp.noise_tol

    def forward(self, x):
        if self.hp.activation == 'linear':
            out = self.input_layer(x)
            out = self.hidden_layer(out)
            out = self.output_layer(out)
        else:
            out = self.act(self.input_layer(x))
            out = self.act(self.hidden_layer(out))
            out = self.output_layer(out)
            if self.hp.task == 'classification':
                out = self.softmax(out)
        return out

    def log_prior(self):
        # Log prior over all the layers
        # return self.input_layer.log_prior + self.output_layer.log_prior
        return self.input_layer.log_prior + self.output_layer.log_prior + self.hidden_layer.log_prior

    def log_post(self):
        # Log posterior over all the layers
        # return self.input_layer.log_post + self.output_layer.log_post
        return self.input_layer.log_post + self.output_layer.log_post + self.hidden_layer.log_post

    def sample_elbo(self, input, target):
        samples = self.hp.n_samples

        # Initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        if self.hp.task == 'regression':
            log_likes = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self(input).squeeze(1)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()

            if self.hp.task == 'regression':
                log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                    target.reshape(-1)).sum()

        if self.hp.task == 'classification':
            log_likes = F.nll_loss(outputs.mean(0), target, reduction='sum')

        # Monte Carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        loss = (1. / self.hp.n_train_batches) * (log_post -
                                                 log_prior) - log_like

        return loss


''' Noise Contrastive Prior
class BNNNCP(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.l1 = nn.Linear(hp.n_input, hp.hidden_units)
        self.l2 = nn.Linear(hp.hidden_units, hp.hidden_units)
        self.lb = BNNLayer(hp.hidden_units, hp.n_output, hp)

        if hp.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif hp.activation == 'relu':
            self.act = nn.ReLU()
        elif hp.activation == 'tanh':
            self.act = nn.Tanh()

        if hp.task == 'classification':
            self.softmax = nn.Softmax()

        self.noise_tol = hp.noise_tol

    def forward(self, x):
        out = self.act(self.l1(x))
        out = self.act(self.l2(out))
        out = self.lb(out)
        return out

    def log_prior(self):
        # return self.l1.log_prior + self.l2.log_prior + self.lb.log_prior
        return self.lb.log_prior

    def log_post(self):
        # return self.l1.log_post + self.l2.log_post + self.lb.log_post
        return self.lb.log_post

    def sample_elbo(self, input, target):
        samples = self.hp.n_samples

        outputs = torch.zeros(samples, target.shape[0])
        log_likes = torch.zeros(samples)

        # Sampling
        for i in range(samples):
            outputs[i] = self(input).squeeze(1)
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()

        # Monte Carlo estimate
        log_like = log_likes.mean()

        # criterion = NLLLoss()
        # nll = criterion(outputs, target)

        return log_like

    def ncp_mean_dist(self, ood_x, y):
        samples = self.hp.pred_samples

        outputs = torch.zeros(samples, y.shape[0])
        for i in range(samples):
            outputs[i] = self(ood_x).squeeze(1)

        # output_prior = Normal(y, self.hp.sigma_y)
        # mean_dist = Normal(outputs.mean(axis=1), outputs.std(axis=1))

        # criterion = nn.KLDivLoss()
        # loss = criterion(output_prior, mean_dist)
        # print(type(loss))
        # kl = kl_divergence(output_prior, mean_dist)
        # print(kl)

        # return kl
        return outputs


class BNNNCP(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.input_layer = BNNLayer(hp.n_input, hp.hidden_units, hp)

        self.hidden_layer = BNNLayer(hp.hidden_units, hp.hidden_units, hp)

        if hp.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif hp.activation == 'relu':
            self.act = nn.ReLU()
        elif hp.activation == 'tanh':
            self.act = nn.Tanh()

        self.output_layer = BNNLayer(hp.hidden_units, hp.n_output, hp)

        if hp.task == 'classification':
            self.softmax = nn.Softmax()

        self.noise_tol = hp.noise_tol  # Used to calculate likelihood

    def forward(self, x):
        out = self.act(self.input_layer(x))
        out = self.act(self.hidden_layer(out))
        out = self.output_layer(out)
        if self.hp.task == 'classification':
            out = self.softmax(out)
        return out

    def forward_ncp(self, x):
        out = self.act(self.input_layer(x.float()))
        self.mean_dist_fn(self.input_layer, x.float())

        out = self.act(self.hidden_layer(out))
        self.mean_dist_fn(self.hidden_layer, out)

        out = self.output_layer(out)

        return self.mean_dist_fn(self.output_layer, out)

    def mean_dist_fn(self, layer, input):
        bias_mean = layer.b_post.mean()
        m = layer.w_post.mean()
        s = layer.w_post.stddev()

        mu_mean = torch.matmul(input, m) + bias_mean
        mu_var = torch.matmul(input**2, s**2)
        mu_std = torch.sqrt(mu_var)

        return Normal(mu_mean, mu_std)

    def log_prior(self):
        # Log prior over all the layers
        # return self.input_layer.log_prior + self.output_layer.log_prior
        return self.input_layer.log_prior + self.output_layer.log_prior + self.hidden_layer.log_prior

    def log_post(self):
        # Log posterior over all the layers
        # return self.input_layer.log_post + self.output_layer.log_post
        return self.input_layer.log_post + self.output_layer.log_post + self.hidden_layer.log_post

    def nll(self, input, target):
        samples = self.hp.n_samples

        # Initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_likes = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self(input).squeeze(1)
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()

        log_like = log_likes.mean()

        return -log_like
'''