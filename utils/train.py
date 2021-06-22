from ..eval.eval_reg import eval_like, eval_mse, eval_reg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .plot import *
import matplotlib.pyplot as plt
import imageio


def train_bnn(net, x_train, y_train, x_test, y_true, hp):
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

            loss = net.sample_elbo(X, y)
            losses.append(loss.data.numpy())

            loss.backward()
            optimizer.step()

        # Evaluation
        with torch.no_grad():
            predictions = net(torch.from_numpy(
                np.array(x_train)).float()).data.numpy()

            # _, predictions, _, = eval_reg(net, x_test, hp.seval_samples)

            mse = eval_mse(predictions, y_train)
            # like = eval_like(net, x_test, y_true, hp)

        loss_lst.append(np.mean(losses))
        mse_lst.append(mse)
        # like_lst.append(sum(like))

        if e % 10 == 0:
            print('epoch: {}'.format(e + 1), 'loss', np.mean(losses), 'MSE',
                  mse)#, 'likelihood', sum(like))

            if e > 5400:
                if hp.plot_progress:
                    image = plot_train_gif(fig, ax, x_train, y_train,
                                           predictions, losses, e)

                    my_images.append(image)

    if hp.plot_progress:
        imageio.mimsave('./train_progress.gif', my_images, fps=9)

    # plot_loss(like)

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
