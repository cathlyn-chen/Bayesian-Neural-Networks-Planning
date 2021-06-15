import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def train_bnn(net, train_data, train_label, hp):
    optimizer = optim.Adam(net.parameters(), lr=hp.learning_rate)

    for e in range(hp.n_epochs):
        losses = []
        for b in range(hp.n_train_batches):
            net.zero_grad()
            X = torch.tensor([
                float(data) for data in train_data
            ][b * hp.batch_size:(b + 1) * hp.batch_size]).reshape(-1, 1)
            y = torch.tensor([float(label) for label in train_label
                              ][b * hp.batch_size:(b + 1) * hp.batch_size])
            loss = net.sample_elbo(X, y)
            # print(type(loss), loss.data)
            losses.append(loss.data.numpy())
            loss.backward()
            optimizer.step()

        if e % 10 == 0:
            print('epoch: {}'.format(e + 1), 'loss', np.mean(losses))
    print('Finished Training')


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
