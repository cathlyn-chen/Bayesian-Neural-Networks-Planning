import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(object):
    def __init__(self, mu, rho, hparams):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.hparams = hparams

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.hparams.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) -
                ((input - self.mu)**2) / (2 * self.sigma**2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.gaussian1 = torch.distributions.Normal(0, hparams.sigma1)
        self.gaussian2 = torch.distributions.Normal(0, hparams.sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.hparams.pi * prob1 +
                          (1 - self.hparams.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, hparams):
        super().__init__()
        self.hparams = hparams
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, hparams)
        # Bias parameters
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, hparams)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.hparams)
        self.bias_prior = ScaleMixtureGaussian(self.hparams)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self, input_size, output_size, hparams):
        super().__init__()
        self.input_size = input_size
        self.l1 = BayesianLinear(input_size, 400, hparams)
        self.l2 = BayesianLinear(400, 400, hparams)
        self.l3 = BayesianLinear(400, 400, hparams)
        self.l4 = BayesianLinear(400, output_size, hparams)
        self.hparams = hparams

    def forward(self, x, sample=False):
        # print(type(x))
        x = x.view(-1, self.input_size)
        # x = np.array(x).reshape((-1, 28 * 28))
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = F.log_softmax(self.l4(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior + self.l3.log_prior + self.l4.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior + self.l4.log_variational_posterior

    def sample_elbo(self, input, target):
        samples = self.hparams.samples
        outputs = torch.zeros(samples, self.hparams.batch_size,
                              self.hparams.classes).to(self.hparams.device)
        log_priors = torch.zeros(samples).to(self.hparams.device)
        log_variational_posteriors = torch.zeros(samples).to(
            self.hparams.device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0),
                                             target,
                                             size_average=False)
        loss = (log_variational_posterior -
                log_prior) / self.hparams.num_batch + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood
