import torch.optim as optim
from torch.distributions import Normal

from .hparams import *
from .utils.train import *
from .data.regression import *
from .data.mnist import *
from .eval.eval_class import *
from .eval.eval_reg import eval_reg

# from .models.bnn1 import BNN
from .models.bnn3 import BNN
from .models.nn import NN
from .data.regression import MoG


def run1():
    hp = mnist_hp()

    net = BNN(hp)
    train_data, train_label, test_data, test_label = get_mnist()
    # print(len(train_data))
    hp.n_train_batches = int(len(train_data) / hp.batch_size)

    optimizer = optim.Adam(net.parameters())
    losses = []
    for epoch in range(hp.n_epochs):
        loss = train1(net, optimizer, train_data, train_label, hp)
        losses.append(loss)

        acc, err = test1(net, test_data, test_label, hp)

        print('epoch', epoch, 'loss', loss, 'test_acc', acc)

    plot_loss(losses)


def run2(net, train_loader, test_loader, hp):
    optimizer = optim.Adam(net.parameters())

    losses = []
    for epoch in range(hp.n_epochs):
        loss = train2(net, optimizer, epoch, train_loader, hp)
        losses.append(loss)

        acc = test2(net, test_loader, hp)
        # test_ensemble(net, test_loader, hp)

        print('epoch', epoch, 'loss', loss.data.numpy(), 'test_acc', acc)


def web_reg():
    hp = reg_hp()
    net = BNN(hp)
    x = torch.tensor([-2, -1.8, -1, 1, 1.8, 2]).reshape(-1, 1)
    y = toy_function(x)
    train_bnn(net, x, y, hp)
    samples = 100
    x_tmp = torch.linspace(-5, 5, 100).reshape(-1, 1)
    y_samp = np.zeros((samples, 100))
    for s in range(samples):
        y_tmp = net(x_tmp).detach().numpy()
        y_samp[s] = y_tmp.reshape(-1)

    plt.plot(x_tmp.numpy(),
             np.mean(y_samp, axis=0),
             label='Mean Posterior Predictive')
    plt.fill_between(x_tmp.numpy().reshape(-1),
                     np.percentile(y_samp, 2.5, axis=0),
                     np.percentile(y_samp, 97.5, axis=0),
                     alpha=0.25,
                     label='95% Confidence')
    plt.legend()
    plt.scatter(x, toy_function(x))
    plt.title('Posterior Predictive')
    plt.show()

    samples = 100
    x_tmp = torch.linspace(-100, 100, 1000).reshape(-1, 1)
    y_samp = np.zeros((samples, 1000))
    for s in range(samples):
        y_tmp = net(x_tmp).detach().numpy()
        y_samp[s] = y_tmp.reshape(-1)
    plt.plot(x_tmp.numpy(),
             np.mean(y_samp, axis=0),
             label='Mean Posterior Predictive')
    plt.fill_between(x_tmp.numpy().reshape(-1),
                     np.percentile(y_samp, 2.5, axis=0),
                     np.percentile(y_samp, 97.5, axis=0),
                     alpha=0.25,
                     label='95% Confidence')
    plt.legend()
    plt.scatter(x, toy_function(x))
    plt.title('Posterior Predictive')
    plt.show()


def run_nn():
    hp = reg_hp()
    net = NN(hp)
    train_data, train_label, x_test, y_true = MoG_data(hp)

    train_data = Variable(torch.from_numpy(np.array(train_data)))
    train_label = Variable(torch.from_numpy(np.array(train_label)))

    train_data = train_data.float()
    train_label = train_label.float()

    train_nn(net, train_data, train_label, hp)
    y_pred = (net((torch.tensor(x_test)).float())).detach().numpy()
    pred_plot(train_data, train_label, x_test, y_pred, y_true)


def run_reg():
    hp = reg_hp()

    # train_data, train_label, x_test, y_true = toy_reg_data(hp)
    # train_data, train_label, x_test, y_true = MoG_data(hp)
    # train_data, train_label, x_test, y_true = paper_reg_data(hp)
    # train_data, train_label, x_test, y_true = f_data(hp)
    # train_data, train_label, x_test, y_true = poly_data(hp)

    train_data, train_label, val_data, val_label, x_test, y_true = MoG_data_val(
        hp)

    # initial_plot(train_data, train_label, x_test, y_true)

    scaler, train_data = transform_data(train_data)
    x_test = scaler.transform(x_test).reshape(-1, 1)

    # train_data = Variable(torch.from_numpy(np.array(train_data)).float())
    # train_label = Variable(torch.from_numpy(np.array(train_label)).float())

    net = BNN(hp)
    losses, mses = train_bnn(net, train_data, train_label, x_test, y_true, hp)

    _, pred_mean, pred_std = eval_reg(net, x_test, hp.pred_samples)

    # train_data = inverse_data(scaler, train_data)
    # x_test = inverse_data(scaler, x_test)

    # pred_plot(train_data, train_label, x_test, pred_mean.reshape(-1, 1),
    #           y_true)

    plt_name = 'mog_unif.png'

    uncertainty_plot(train_data, train_label, x_test, y_true, pred_mean,
                     pred_std, plt_name)

    # plot_hist(net)

    # web(net)


def run_reg_2d():
    hp = reg_2d_hp()

    # x_train, y_train, x_test, y_true = gaussian_data_2d(hp)

    x_train, y_train, x_val, y_val, x_test, y_true = poly_data_2d(hp)

    # init_plot_contour(x_train, x_val, y_true, hp)
    # init_plot_3d(x_train, y_train, x_val, y_val, y_true, hp)

    # init_plot_3d(x_test, y_true, y_true, hp)

    x_train = Variable(torch.Tensor(x_train))
    y_train = Variable(torch.Tensor(y_train))

    net = BNN(hp)

    # Standardize
    scaler_x, x_train = transform_data(x_train)
    scaler_y, y_train = transform_data(y_train.reshape(-1, 1))

    x_test = scaler_x.transform(x_test)
    y_true = scaler_y.transform(y_true.reshape(-1, 1))

    losses, mses = train_bnn(net, x_train, y_train, x_test, y_true, hp)

    plot_loss(losses)
    plot_loss(mses)

    # y_pred = (net((torch.tensor(x_train)).float())).detach().numpy()

    _, y_pred, pred_std = eval_reg(net, x_test, hp.pred_samples)
    # print(pred_std.shape)

    # Back to original data
    x_train = inverse_data(scaler_x, x_train)
    x_test = inverse_data(scaler_x, x_test)
    y_train = inverse_data(scaler_y, y_train)
    y_pred = inverse_data(scaler_y, y_pred)
    y_true = inverse_data(scaler_y, y_true)

    # Uncertainty contour plot
    uncertainty_plot_contour(x_train, pred_std, hp)

    # Prediction plot
    pred_plot_3d(x_train, y_train, x_test, y_pred, y_true, hp)


def run_nn_2d():
    hp = reg_2d_hp()
    net = NN(hp)
    x_train, y_train, x_val, y_val, x_test, y_true = poly_data_2d(hp)

    x_train = Variable(torch.Tensor(x_train)).float()
    y_train = Variable(torch.Tensor(y_train)).float()

    train_nn(net, x_train, y_train, hp)
    y_pred = (net((torch.Tensor(x_test)).float())).detach().numpy()

    # init_plot_3d(x_test, y_pred, y_true, hp)
    pred_plot_3d(x_train, y_train, x_test, y_pred, y_true, hp)


if __name__ == '__main__':
    # run1()

    # run_reg()
    # run_nn()

    run_reg_2d()
    # run_nn_2d()
