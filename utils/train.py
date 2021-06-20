from ..eval.eval_reg import eval_reg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .plot import *
import matplotlib.pyplot as plt
import imageio


def train_bnn(net, train_data, train_label, x_test, y_true, hp):
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    if hp.plot_progress:
        my_images = []
        fig, ax = plt.subplots()

    loss_lst = []
    mse_lst = []

    for e in range(hp.n_epochs):
        losses = []

        for b in range(hp.n_train_batches):
            net.zero_grad()
            X = Variable(
                torch.from_numpy(
                    np.array(train_data[b * hp.batch_size:(b + 1) *
                                        hp.batch_size])).float().reshape(
                                            -1, 1))
            y = Variable(
                torch.from_numpy(
                    np.array(train_label[b * hp.batch_size:(b + 1) *
                                         hp.batch_size])).float())
            loss = net.sample_elbo(X, y)

            losses.append(loss.data.numpy())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = net(torch.from_numpy(
                np.array(train_data)).float()).data.numpy()

            mse = (np.square(predictions - np.array(train_label))).mean()

        # _, pred_test, _, = eval_reg(net, x_test)

        loss_lst.append(np.mean(losses))
        mse_lst.append(mse)

        if e % 10 == 0:
            print('epoch: {}'.format(e + 1), 'loss', np.mean(losses), 'MSE',
                  mse)
                  
            if e > 5400:
                if hp.plot_progress:
                    image = plot_train_gif(fig, ax, train_data, train_label,
                                           predictions, losses, x_test, y_true,
                                           e)

                    my_images.append(image)

    if hp.plot_progress:
        imageio.mimsave('./train_progress.gif', my_images, fps=9)

    if hp.plot_loss:
        plot_loss(loss_lst)

    print('Finished Training')


def train_nn(net, train_data, train_label, hp):
    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    for epoch in range(hp.n_epochs):
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label)
        if epoch % 10 == 0:
            print("epoch {} MSE: {}".format(epoch, loss))
        loss.backward()
        optimizer.step()


def train1(net, optimizer, train_data, train_label, hp):
    losses = []
    for b in range(hp.num_batch):
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
