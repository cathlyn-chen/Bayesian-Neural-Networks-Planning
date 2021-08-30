from torch.distributions.kl import kl_divergence
from torch.nn.functional import kl_div
import torch.optim as optim
from torch.distributions import Normal

from .hparams import *
from .utils.train import *
from .utils.plot import *
from .data.regression import *
from .data.navigation import *
from .data.mnist import *
from .eval.eval_class import *
from .eval.eval_reg import *

# from .models.bnn1 import BNN
from .models.bnn3 import *
from .models.nn import NN


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


def run_nn():
    hp = reg_hp()
    net = NN(hp)
    train_data, train_label, x_test, y_true = ncp_data(hp)

    train_data = Variable(torch.from_numpy(np.array(train_data)))
    train_label = Variable(torch.from_numpy(np.array(train_label)))

    train_data = train_data.float()
    train_label = train_label.float()

    train_nn(net, train_data, train_label, hp)
    y_pred = (net((torch.tensor(x_test)).float())).detach().numpy()
    pred_plot(train_data, train_label, x_test, y_pred, y_true)


def run_reg():
    hp = reg_hp()

    # Different ground truth functions
    # x_train, y_train, x_test, y_true = MoG_data(hp)
    # x_train, y_train, x_test, y_true = paper_reg_data(hp)
    # x_train, y_train, x_test, y_true = f_data(hp)
    # x_train, y_train, x_test, y_true = poly_data(hp)
    x_train, y_train, x_test, y_true = ncp_data(hp)

    # x_train, y_train, x_val, y_val, x_test, y_true = toy_reg_data(hp)
    # x_train, y_train, x_val, y_val, x_test, y_true = MoG_data_val(hp)

    initial_plot(x_train, y_train, x_test, y_true)

    scaler, x_train = transform_data(x_train)
    x_test = scaler.transform(x_test).reshape(-1, 1)

    # train_data = Variable(torch.from_numpy(np.array(train_data)).float())
    # train_label = Variable(torch.from_numpy(np.array(train_label)).float())

    net = BNN(hp)
    losses, mses, likes = train_bnn(net, x_train, y_train, x_test, y_true, hp)
    # losses, mses, likes = train_bnn(net, x_train, y_train, x_val, y_val, hp)

    plot_loss(losses)
    plot_loss(mses)
    # plot_loss(likes)

    _, pred_mean, pred_std = eval_reg(net, x_test, hp.pred_samples)

    x_train = inverse_data(scaler, x_train)
    x_test = inverse_data(scaler, x_test)

    # pred_plot(train_data, train_label, x_test, pred_mean.reshape(-1, 1),
    #           y_true)

    plt_name = 'plt.png'

    uncertainty_plot(x_train, y_train, x_test, y_true, pred_mean, pred_std,
                     plt_name)

    # plot_hist(net)


def run_reg_2d():
    hp = reg_2d_hp()

    # x_train, y_train, x_test, y_true = gaussian_data_2d(hp)
    x_train, y_train, x_val, y_val, x_test, y_true = poly_data_2d(hp)

    # Initial plots
    # init_plot_contour(x_train, x_val, y_true, hp)
    # init_plot_3d(x_train, y_train, x_val, y_val, y_true, hp)

    x_train = Variable(torch.Tensor(x_train))
    y_train = Variable(torch.Tensor(y_train))
    x_val = Variable(torch.Tensor(x_val))
    y_val = Variable(torch.Tensor(y_val))

    net = BNN(hp)

    # Standardize data
    scaler_x, x_train = transform_data(x_train)
    scaler_y, y_train = transform_data(y_train.reshape(-1, 1))

    x_test = scaler_x.transform(x_test)
    y_true = scaler_y.transform(y_true.reshape(-1, 1))

    # Train BNN
    losses, mses, likes = train_bnn(net, x_train, y_train, x_val, y_val, hp)

    # Loss plots
    # plot_loss(losses)
    # plot_loss(mses)
    # plot_loss(likes)

    _, y_pred, pred_std = eval_reg(net, x_test, hp.pred_samples)

    like = eval_like(net, x_test, y_true, hp)

    # Back to original data
    x_train = inverse_data(scaler_x, x_train)
    x_test = inverse_data(scaler_x, x_test)
    y_train = inverse_data(scaler_y, y_train)
    y_pred = inverse_data(scaler_y, y_pred)
    y_true = inverse_data(scaler_y, y_true)

    # Likelihood contour plot for test data
    plot_like_contour(x_train, like, hp)

    # Uncertainty contour plot
    uncertainty_plot_contour(x_train, pred_std, hp)

    # Prediction plot
    pred_plot_3d(x_train, y_train, y_pred, y_true, hp)


def run_nn_2d():
    hp = reg_2d_hp()
    net = NN(hp)
    x_train, y_train, x_val, y_val, x_test, y_true = poly_data_2d(hp)

    x_train = Variable(torch.Tensor(x_train)).float()
    y_train = Variable(torch.Tensor(y_train)).float()

    losses, mses, likes = train_nn(net, x_train, y_train, x_val, y_val, hp)
    y_pred = (net((torch.Tensor(x_test)).float())).detach().numpy()

    # init_plot_3d(x_test, y_pred, y_true, hp)
    pred_plot_3d(x_train, y_train, y_pred, y_true, hp)


''' Noise Contrastive Prior 
def run_ncp():
    hp = reg_ncp()

    x_train, y_train, x_test, y_true = poly_data(hp)
    # x_train, y_train, x_test, y_true = ncp_data(hp)

    # initial_plot(x_train, y_train, x_test, y_true)

    scaler, x_train = transform_data(x_train)
    x_test = scaler.transform(x_test).reshape(-1, 1)

    x_train = Variable(torch.from_numpy(np.array(x_train)).float())
    y_train = Variable(torch.from_numpy(np.array(y_train)).float())

    net = BNNNCP(hp)
    losses, mses = train_bnn_ncp(net, x_train, y_train, x_test, y_true, hp)

    plot_loss(losses)
    plot_loss(mses)

    _, pred_mean, pred_std = eval_reg(net, x_test, hp.pred_samples)

    x_train = inverse_data(scaler, x_train)
    x_test = inverse_data(scaler, x_test)

    # pred_plot(train_data, train_label, x_test, pred_mean.reshape(-1, 1),
    #           y_true)

    plt_name = 'mog_unif.png'

    uncertainty_plot(x_train, y_train, x_test, y_true, pred_mean, pred_std,
                     plt_name)

    # plot_hist(net)
'''


def run_nav():
    hp = nav_hp()
    x1, x2, y1, y2 = nav_sample(hp)

    train_data_x, train_label_x, train_data_y, train_label_y, test_data_x, test_data_y, test_label_x, test_label_y = nav_data_pair(
        x1, x2, y1, y2, hp)

    # nav_plot(x1, x2, y1, y2, hp)
    nav_plot(x1, x1 + x2, y1, y1 + y2, hp)

    # Standardize data
    scaler_data_x, train_data_x = transform_data(train_data_x)
    scaler_label_x, train_label_x = transform_data(train_label_x.reshape(
        -1, 1))

    test_data_x = scaler_data_x.transform(test_data_x)
    test_label_x = scaler_label_x.transform(test_label_x.reshape(-1, 1))

    train_data_x = Variable(torch.Tensor(train_data_x).reshape(-1, 2).float())
    train_label_x = Variable(
        torch.Tensor(train_label_x).reshape(-1, 1).float())
    '''x direction'''

    net_x = BNN(hp)
    _, _, _ = train_bnn(net_x, train_data_x, train_label_x, test_data_x,
                        test_label_x, hp)

    # plot_loss(losses)
    # plot_loss(mses)
    # plot_loss(likes)

    _, pred_mean_x, pred_std_x = eval_reg(net_x, test_data_x, hp.pred_samples)
    '''y direction'''

    # Standardize data
    scaler_data_y, train_data_y = transform_data(train_data_y)
    scaler_label_y, train_label_y = transform_data(train_label_y.reshape(
        -1, 1))

    test_data_y = scaler_data_y.transform(test_data_y)
    test_label_y = scaler_label_y.transform(test_label_y.reshape(-1, 1))

    train_data_y = Variable(torch.Tensor(train_data_y).reshape(-1, 2).float())
    train_label_y = Variable(
        torch.Tensor(train_label_y).reshape(-1, 1).float())
    # print(train_data_x.shape, train_label_x.shape)

    net_y = BNN(hp)
    _, mse_lst, _ = train_bnn(net_y, train_data_y, train_label_y, test_data_y,
                              test_label_y, hp)

    _, pred_mean_y, pred_std_y = eval_reg(net_y, test_data_y, hp.pred_samples)

    print(
        test_label_x[:9],
        pred_mean_x.reshape(-1, 1)[:9],
    )
    print(test_label_y[:9], pred_mean_y.reshape(-1, 1)[:9])

    plot_loss(mse_lst)

    pred_std = pred_std_x + pred_std_y

    # Back to original data
    # x_train = inverse_data(scaler_data_x, x_train)
    # x_test = inverse_data(scaler_x, x_test)
    # y_train = inverse_data(scaler_y, y_train)
    # y_pred = inverse_data(scaler_y, y_pred)
    # y_true = inverse_data(scaler_y, y_true)

    # train_data_x = inverse_data(scaler_data_x, train_data_x)
    # train_label_x = inverse_data(scaler_label_x, train_label_x)

    # MSEs
    mse_x = eval_mse(pred_mean_x, test_label_x)
    mse_y = eval_mse(pred_mean_y, test_label_y)
    mse = mse_x + mse_y

    # print(mse_x.shape, mse_y.shape, mse.shape)

    nav_pred_plot(x1, x2, y1, y2, pred_mean_x + pred_mean_y,
                  test_label_x + test_label_y, hp)
    nav_uncertainty(x1, x1 + x2, y1, y1 + y2, mse, hp)
    nav_uncertainty(x1, x1 + x2, y1, y1 + y2, pred_std, hp)


def run_nav_nn():
    hp = nav_hp()
    x1, x2, y1, y2 = nav_sample(hp)

    train_data_x, train_label_x, train_data_y, train_label_y, test_data_x, test_data_y, test_label_x, test_label_y = nav_data_pair(
        x1, x2, y1, y2, hp)

    nav_plot(x1, x2, y1, y2, hp)

    train_data_x = Variable(torch.Tensor(train_data_x).reshape(-1, 2).float())
    train_label_x = Variable(
        torch.Tensor(train_label_x).reshape(-1, 1).float())
    # print(train_data_x.shape, train_label_x.shape)
    '''x direction'''

    net_x = NN(hp)
    train_nn(net_x, train_data_x, train_label_x, hp)

    pred_x = (net_x((torch.tensor(test_data_x)).float())).detach().numpy()

    print(test_label_x.shape, pred_x.shape)
    '''y direction'''

    train_data_y = Variable(torch.Tensor(train_data_y).reshape(-1, 2).float())
    train_label_y = Variable(
        torch.Tensor(train_label_y).reshape(-1, 1).float())
    # print(train_data_x.shape, train_label_x.shape)

    net_y = NN(hp)
    train_nn(net_y, train_data_y, train_label_y, hp)

    pred_y = (net_y((torch.tensor(test_data_y)).float())).detach().numpy()

    print(test_label_y.shape, pred_y.shape)


if __name__ == '__main__':
    # run_reg()
    # run_nn()

    # run_reg_2d()
    # run_nn_2d()

    # run_ncp()
    run_nav()
    # run_nav_nn()