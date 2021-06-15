import torch.optim as optim

from .hparams import reg_hp
from .utils.train import *
from .data.regression import *
from .data.mnist import *
from .eval.eval_class import *
from .eval.eval_reg import eval_reg

from .models.bnn1 import BNN
# from .models.bnn3 import BNN


def run1(net, train_data, train_label, test_data, test_label, hp):
    optimizer = optim.Adam(net.parameters())
    losses = []
    for epoch in range(hp.n_epochs):
        loss = train1(net, optimizer, train_data, train_label, hp)
        losses.append(loss)

        acc, err = test1(net, test_data, test_label, hp)

        print('epoch', epoch, 'loss', loss, 'test_acc', acc)

    return losses


def run2(net, train_loader, test_loader, hp):
    optimizer = optim.Adam(net.parameters())

    losses = []
    for epoch in range(hp.n_epochs):
        loss = train2(net, optimizer, epoch, train_loader, hp)
        losses.append(loss)

        acc = test2(net, test_loader, hp)
        # test_ensemble(net, test_loader, hp)

        print('epoch', epoch, 'loss', loss.data.numpy(), 'test_acc', acc)


def run_reg():
    hp = reg_hp()
    # train_data, train_label, x_test, y_true = toy_reg_data(hp)
    train_data, train_label, x_test, y_true = MoG_data(hp)
    # train_data, train_label, x_test, y_true = paper_reg_data(hp)
    # train_data, train_label, x_test, y_true = f_data(hp)
    # train_data, train_label, x_test, y_true = poly_data(hp)

    # train_data, train_label, val_data, val_label, x_test, y_true = MoG_data_unif(hp)

    # initial_plot(train_data, train_label, x_test, y_true)

    net = BNN(hp)
    train_bnn(net, train_data, train_label, hp)

    pred_lst, pred_mean, pred_std = eval_reg(x_test, net)

    pred_plot(train_data, train_label, x_test, y_true, pred_lst, pred_mean,
              pred_std)

    # plot_hist(net)

    # web(net)


if __name__ == '__main__':
    run_reg()