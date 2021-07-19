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

    plt_name = 'mog_unif.png'

    uncertainty_plot(x_train, y_train, x_test, y_true, pred_mean, pred_std,
                     plt_name)

    # plot_hist(net)

    # web(net)


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

    # y_pred = (net((torch.tensor(x_train)).float())).detach().numpy()

    _, y_pred, pred_std = eval_reg(net, x_test, hp.pred_samples)

    like = eval_like(net, x_test, y_true, hp)
    # print(like)

    # print(y_pred.shape, type(likes[0]))

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


def test_log_prob():
    dist = Normal(3, 1)
    sample = dist.sample()  # x
    lp = dist.log_prob(sample)
    print(
        lp,
        sample)  # more likly the sample (closer to mean), greater the log_prob


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


def run_nav():
    hp = nav_hp()
    nav_data(hp)


if __name__ == '__main__':
    # run1()

    # run_reg()
    # run_nn()
    # run_ncp()

    run_reg_2d()
    # run_nn_2d()
    '''
    # test_log_prob()
    p1 = Normal(0, 3)
    p2 = Normal(0, 1)
    v1 = p1.sample((900, ))
    v2 = p2.sample((900, ))
    print(kl_divergence(p1, p2))
    print(kl_div(v1, v2))
    '''