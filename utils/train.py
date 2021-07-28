import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import kl_div
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal, kl_divergence

import matplotlib.pyplot as plt
import imageio
from scipy.stats import norm
import seaborn as sns

from .plot import *
from .tools import *
from ..eval.eval_reg import *


def train_bnn(net, x_train, y_train, x_val, y_val, hp):
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    # Plot training progress gif
    if hp.plot_progress:
        my_images = []
        fig, ax = plt.subplots()

    loss_lst = []
    mse_lst = []
    like_lst = []

    for e in range(hp.n_epochs):
        losses = []

        # Minibathces
        for b in range(hp.n_train_batches):
            net.zero_grad()
            X = Variable(
                torch.Tensor(x_train[b * hp.batch_size:(b + 1) *
                                     hp.batch_size]).float())
            y = Variable(
                torch.Tensor(y_train[b * hp.batch_size:(b + 1) *
                                     hp.batch_size])).float()

            # Forward sample to return loss
            loss = net.sample_elbo(X, y)
            losses.append(loss.data.numpy())

            loss.backward()
            optimizer.step()

        # Evaluation
        with torch.no_grad():
            predictions = net(torch.from_numpy(
                np.array(x_val)).float()).data.numpy()

            # _, predictions, _, = eval_reg(net, x_test, hp.val_samples)

            mse = eval_mse(predictions, y_val).mean()
            # print(x_val.shape, y_val.shape)
            # like = eval_like(net, x_val, y_val, hp)
            # print(like.shape, type(like))
            # print(x_train.shape, predictions.shape)

        loss_lst.append(np.mean(losses))
        mse_lst.append(mse)
        # like_lst.append(np.sum(like))
        #sum(like).data.numpy()[0]

        if e % 10 == 0:
            print('epoch: {}'.format(e + 1), 'loss', np.mean(losses), 'MSE',
                  mse)  #, 'likelihood', np.sum(like))

            if e > 5400:
                if hp.plot_progress:
                    image = plot_train_gif(fig, ax, x_train, y_train,
                                           predictions, losses, e)

                    my_images.append(image)

    if hp.plot_progress:
        imageio.mimsave('./train_progress.gif', my_images, fps=9)

    # plot_like(x_val, like)

    # plot_like_3d(x_val, like)

    print('Finished Training')

    return loss_lst, mse_lst, like_lst


def train_bnn_ncp(net, x_train, y_train, x_val, y_val, hp):
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    loss_lst = []
    mse_lst = []

    for e in range(hp.n_epochs):
        losses = []

        X = Variable(torch.Tensor(x_train).float())

        ood_X = (X + np.random.normal(0, hp.sigma_x, size=X.shape)).float()

        # input_prior = np.sum(norm(0, hp.sigma_x).pdf(ood_X - X),
        #                      axis=0) / X.shape[0]
        # input_prior = np.mean(norm(0, hp.sigma_x).pdf(ood_X - X), axis=1)
        input_prior = Normal(ood_X - X, hp.sigma_x)
        # print(ood_X.shape, input_prior)

        # plt.scatter(ood_X.reshape(-1, 1), input_prior)
        # sns.lineplot(x=ood_X, y=input_prior)
        # plt.show()

        y = Variable(torch.Tensor(y_train).float())
        output_prior = Normal(y, hp.sigma_y)

        hp.data_prior = input_prior * output_prior

        log_like = net.sample_elbo(X, y)

        ood_mean = net.ncp_mean_dist(ood_X, y)
        ood_y = output_prior.sample(
            (hp.pred_samples, )).reshape(hp.pred_samples, -1)

        # kl = net.ncp_mean_dist(ood_X, y)
        # loss = kl - log_like

        # print(ood_mean.shape, ood_y.shape)

        # np.mean(norm(0, hp.sigma_x).pdf(ood_X - X), axis=1)

        # loss = kl_divergence(output_prior, ood_mean) - log_like
        loss = kl_div(ood_y, ood_mean) - log_like

        losses.append(loss.data.numpy())

        loss.backward()
        optimizer.step()
        '''
        # Minibathces
        for b in range(hp.n_train_batches):
            net.zero_grad()
            X = Variable(
                torch.Tensor(x_train[b * hp.batch_size:(b + 1) *
                                     hp.batch_size]).float())

            
            y = Variable(
                torch.Tensor(y_train[b * hp.batch_size:(b + 1) *
                                     hp.batch_size])).float()

            ood_mean_prior = Normal(y, hp.sigma_y)

            ood_mean_dist = net.forward_ncp(ood_X)

            # Loss
            nll = net.nll(X, y)
            kl = nn.MSELoss()
            kl_loss = kl(ood_mean_prior, ood_mean_dist)

            loss = kl_loss + nll
            losses.append(loss.data.numpy())

            loss.backward()
            optimizer.step()
        '''

        # Evaluation
        with torch.no_grad():
            predictions = net(torch.from_numpy(
                np.array(x_val)).float()).data.numpy()
            mse = eval_mse(predictions, y_val).mean()


        loss_lst.append(np.mean(losses))
        mse_lst.append(mse)

        if e % 10 == 0:
            print('epoch: {}'.format(e + 1), 'loss', np.mean(losses), 'MSE',
                  mse)
    # plot_like(x_val, like)

    print('Finished Training')

    return loss_lst, mse_lst


def train_nn(net, train_data, train_label, hp):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    for epoch in range(hp.n_epochs):
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label.reshape(-1, 1))

        # print(train_label.shape)

        if epoch % 10 == 0:
            print("epoch {} MSE: {}".format(epoch, loss))
        loss.backward()
        optimizer.step()


def train1(net, optimizer, train_data, train_label, hp):
    losses = []
    for b in range(hp.n_train_batches):
        net.zero_grad()

        # Obtain minibatch
        X = Variable(
            torch.Tensor(train_data[b * hp.batch_size:(b + 1) *
                                    hp.batch_size]))
        y = Variable(
            torch.Tensor(train_label[b * hp.batch_size:(b + 1) *
                                     hp.batch_size]))

        loss, _, _, _ = net.sample_elbo(X, y)
        loss.backward()
        optimizer.step()

        losses.extend(loss.data.numpy())

    return np.mean(losses)


def train2(net, optimizer, epoch, train_loader, hp):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(hp.device), target.to(hp.device)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(
            data, target)
        loss.backward()
        optimizer.step()

    return loss
