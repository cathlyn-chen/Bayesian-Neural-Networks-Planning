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
        # Set input & output dimensions
        self.n_input = n_input
        self.n_output = n_output

        # Initialize mu and rho parameters for layer's weights
        self.w_mu = nn.Parameter(torch.zeros(n_output, n_input))
        self.w_rho = nn.Parameter(torch.zeros(n_output, n_input))

        # Initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(n_output))
        self.b_rho = nn.Parameter(torch.zeros(n_output))

        # Initialize weight samples - calculated whenever the layer makes a prediction
        self.w = None
        self.b = None

        # Initialize prior distribution for all of the weights and biases
        if hp.prior == 'gaussian':
            self.prior = torch.distributions.Normal(0, hp.sigma_prior1)
        elif hp.prior == 'scale_mixture':
            self.prior = ScaleMixtureGaussian(hp)

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

        return F.linear(input, self.w, self.b)


class BNN(nn.Module):
    def __init__(self, n_input, hidden_units, n_output, hp):
        # Initialize the network but using the BBB layer
        super().__init__()
        self.hp = hp
        self.input = BNNLayer(n_input, hidden_units, hp)

        if hp.activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif hp.activation == 'relu':
            self.act = nn.ReLU()

        self.output = BNNLayer(hidden_units, n_output, hp)

        if hp.task == 'classification':
            self.softmax = nn.Softmax()

        self.noise_tol = hp.noise_tol  # Used to calculate likelihood

    def forward(self, x):
        out = self.input(x)
        out = self.act(out)
        out = self.output(out)
        if self.hp.task == 'classification':
            out = self.softmax(out)
        return out

    def log_prior(self):
        # Log prior over all the layers
        return self.input.log_prior + self.output.log_prior

    def log_post(self):
        # Log posterior over all the layers
        return self.input.log_post + self.output.log_post

    def sample_elbo(self, input, target, samples):
        # Negative elbo as loss function
        # Initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # Make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()
        # Monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # Negative elbo
        loss = log_post - log_prior - log_like
        return loss