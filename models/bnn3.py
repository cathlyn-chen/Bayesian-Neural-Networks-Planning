import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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

        nn.init.xavier_uniform_(self.w_mu)
        nn.init.xavier_uniform_(self.w_rho)
        nn.init.zeros_(self.b_mu)
        nn.init.zeros_(self.b_rho)

        # Initialize weight samples - calculated whenever the layer makes a prediction
        self.w = None
        self.b = None

        # Initialize prior distribution for all of the weights and biases
        if hp.prior == 'gaussian':
            self.prior = torch.distributions.Normal(0, hp.sigma_prior1)
        elif hp.prior == 'scale_mixture':
            self.prior = ScaleMixtureGaussian(hp)
        elif hp.prior == 'ncp':
            pass

    # '''
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
        # print(forward.shape)
        return forward

    # '''
    ''' Sample N in parallel
    def forward(self, input):
        N = self.hp.n_samples

        # Sample weights
        w_epsilon = [Normal(0, 1).sample(self.w_mu.shape)
                     for _ in range(N)]  #(N, (H, I))
        self.w = [
            self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * eps
            for eps in w_epsilon
        ]  #(N, (H, I))

        # Sample bias
        b_epsilon = [Normal(0, 1).sample(self.b_mu.shape)
                     for _ in range(N)]  #(N, (H, I))
        self.b = [
            self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * eps
            for eps in b_epsilon
        ]

        # Log prior - evaluating log pdf of prior at sampled weight and bias
        w_log_prior = [self.prior.log_prob(w) for w in self.w]
        b_log_prior = [self.prior.log_prob(b) for b in self.b]
        self.log_prior = [
            torch.sum(w_log_prior[i]) + torch.sum(b_log_prior[i])
            for i in range(N)
        ]  #(N, 1)

        # Log variational posterior - evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = [
            Normal(self.w_mu.data, torch.log(1 + torch.exp(rho)))
            for rho in self.w_rho
        ]
        self.b_post = [
            Normal(self.b_mu.data, torch.log(1 + torch.exp(rho)))
            for rho in self.b_rho
        ]
        self.log_post = [
            self.w_post[i].log_prob(self.w[i]).sum() +
            self.b_post[i].log_prob(self.b[i]).sum() for i in range(N)
        ]  #(N, 1)

        output = [F.linear(input, self.w[i], self.b[i]) for i in range(N)]
        return output

    '''


class BNN(nn.Module):
    def __init__(self, hp):
        # Initialize the network but using the BBB layer
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

        # ''' For Loop

        # Initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        # print(target.shape)
        # print(outputs.shape)
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        if self.hp.task == 'regression':
            log_likes = torch.zeros(samples)

        # print(input.shape, target.shape)
        # print(self(input).shape)

        for i in range(samples):
            # print(input.shape, self(input).shape)
            outputs[i] = self(input).squeeze(1)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()

            if self.hp.task == 'regression':
                log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                    target.reshape(-1)).sum()

        if self.hp.task == 'classification':
            log_likes = F.nll_loss(outputs.mean(0), target, reduction='sum')

        # '''
        '''
        print(type(input))
        outputs = self(input)
        log_priors = self.log_prior()
        log_posts = self.log_post()
        log_likes = [
            Normal(output,
                   self.noise_tol).log_prob(target.reshape(-1)).sum()
            for output in outputs
        ]
        '''

        # Monte Carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        loss = (1. / self.hp.n_train_batches) * (log_post -
                                                 log_prior) - log_like

        return loss


class BNNNCP(nn.Module):
    def __init__(self, hp):
        # Initialize the network but using the BBB layer
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