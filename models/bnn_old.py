import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gaussian(object):
    def __init__(self, mu, rho, hp):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.hp = hp

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.hp.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) -
                ((input - self.mu)**2) / (2 * self.sigma**2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.gaussian1 = torch.distributions.Normal(0, hp.sigma1)
        self.gaussian2 = torch.distributions.Normal(0, hp.sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.hp.pi * prob1 + (1 - self.hp.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, hp):
        super().__init__()
        self.hp = hp
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, hp)
        # Bias parameters
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, hp)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.hp)
        self.bias_prior = ScaleMixtureGaussian(self.hp)
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
    def __init__(self, input_size, output_size, hidden_units, hp):
        super().__init__()
        self.input_size = input_size
        self.l1 = BayesianLinear(input_size, hidden_units, hp)
        self.l2 = BayesianLinear(hidden_units, hidden_units, hp)
        self.l3 = BayesianLinear(hidden_units, output_size, hp)
        self.hp = hp

    def forward(self, x, sample=False):
        # print(type(x))
        x = x.reshape(-1, self.input_size)
        # x = np.array(x).reshape((-1, 28 * 28))
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior

    def sample_elbo(self, input, target):
        samples = self.hp.samples
        outputs = torch.zeros(samples, self.hp.batch_size,
                              self.hp.classes).to(self.hp.device)
        log_priors = torch.zeros(samples).to(self.hp.device)
        log_variational_posteriors = torch.zeros(samples).to(self.hp.device)
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
                log_prior) / self.hp.num_batch + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood
