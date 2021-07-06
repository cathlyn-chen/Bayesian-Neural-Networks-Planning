import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm


def initial_plot(train_data, train_label, x_test, y_true):
    plt.scatter(train_data,
                train_label,
                marker='+',
                label='Training data',
                color='black')

    plt.plot(x_test, y_true, label='Truth')

    plt.title('Noisy Training Data and Ground Truth')
    plt.legend()
    plt.show()


def init_plot_contour(x_train, x_val, y_true, hp):
    x1, x2 = hp.grid

    fig, ax = plt.subplots()

    # Truth
    con = ax.contourf(x1, x2, y_true.reshape(x1.shape), cmap='viridis')

    # Sampled training data
    ax.scatter(x_train[:, 0],
               x_train[:, 1],
               marker='+',
               color='black',
               alpha=.6,
               label='Train data')

    # Validation data
    ax.scatter(x_val[:, 0],
               x_val[:, 1],
               marker='o',
               color='green',
               alpha=.3,
               label='Validation data')

    # Colour bar
    cbar = plt.colorbar(con)
    cbar.ax.set_ylabel('y value', fontsize=9)

    ax.set_aspect('equal')
    ax.set_title('Ground Truth Contour Plot & Sampled Points')
    plt.legend()
    plt.show()


def uncertainty_plot_contour(x_train, std, hp):
    x1, x2 = hp.grid

    fig, ax = plt.subplots()

    # Truth
    con = ax.contourf(x1, x2, std.reshape(x1.shape), cmap='cividis')

    # Sampled training data points
    ax.scatter(x_train[:, 0],
               x_train[:, 1],
               color='black',
               marker='+',
               alpha=.6,
               label='Train data')

    # Colour bar for uncertainty
    cbar = plt.colorbar(con)
    cbar.ax.set_ylabel('Uncertainty (+/- 3 std)', fontsize=9)

    ax.set_aspect('equal')
    # ax.set_title('Samples from bivariate normal distribution')
    plt.legend()
    plt.show()


def init_plot_3d(x_train, y_train, x_val, y_val, y_true, hp):
    x1, x2 = hp.grid

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Truth
    ax.plot_surface(x1,
                    x2,
                    y_true.reshape(x1.shape),
                    linewidth=0,
                    color='grey',
                    alpha=0.3)

    # Sampled training data points
    ax.scatter(x_train[:, 0],
               x_train[:, 1],
               y_train,
               marker='x',
               c='black',
               alpha=.6,
               label='Train data')

    # Validation data points
    ax.scatter(x_val[:, 0],
               x_val[:, 1],
               y_val,
               marker='o',
               c='green',
               alpha=.3,
               label='Validation data')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    ax.set_title('Ground Truth & Sampled Points')

    plt.legend()
    plt.show()


def pred_plot_3d(x_train, y_train, y_pred, y_true, hp):
    x1, x2 = hp.grid

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Truth
    ax.plot_surface(x1,
                    x2,
                    y_true.reshape(x1.shape),
                    color='grey',
                    linewidth=0,
                    alpha=0.3,
                    label='Truth')
    # cmap='viridis'

    # Pred plot
    ax.plot_surface(x1,
                    x2,
                    np.array(y_pred).reshape((x1.shape)),
                    color='cornflowerblue',
                    alpha=0.6)

    # Sampled data points
    ax.scatter(x_train[:, 0],
               x_train[:, 1],
               y_train,
               marker='x',
               c='black',
               alpha=.6)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    plt.show()


def pred_plot(train_data, train_label, x_test, y_pred, y_true):
    plt.scatter(train_data,
                train_label,
                marker='+',
                label='Training data',
                color='black')

    plt.plot(x_test, y_true, label='Truth', color='grey')
    plt.plot(x_test, y_pred, c='royalblue', label='Pred')
    plt.legend()
    plt.show()


def uncertainty_plot(train_data, train_label, x_test, y_true, pred_mean,
                     pred_std, name):
    plt.plot(x_test, pred_mean, c='royalblue', label='Mean Pred')
    plt.fill_between(x_test.reshape(-1, ),
                     pred_mean - 3 * pred_std,
                     pred_mean + 3 * pred_std,
                     color='cornflowerblue',
                     alpha=.5,
                     label='Epistemic Uncertainty (+/- 3 std)')
    # plt.fill_between(x_test.reshape(-1),
    #                  np.percentile(pred_lst, 2.5, axis=0),
    #                  np.percentile(pred_lst, 97.5, axis=0),
    #                  alpha=0.6,
    #                  color='cornflowerblue',
    #                  label='95% Confidence')

    plt.scatter(train_data,
                train_label,
                marker='+',
                color='black',
                label='Training Data')
    plt.plot(x_test, y_true, color='grey', label='Truth')

    plt.legend()
    # plt.savefig(name)
    plt.show()


def plot_train_gif(fig, ax, train_data, train_label, predictions, losses, e):
    plt.cla()
    ax.set_title('Training Progress', fontsize=21)
    ax.set_xlabel('Data', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    # ax.set_xlim(-0.9, 2.4)
    ax.set_ylim(min(train_label) - 0.3, max(train_label) + 0.45)
    ax.scatter(train_data, train_label, color="black")
    # ax.plot(x_test, y_true, label='Truth', color='grey')
    ax.plot(train_data, predictions, c='royalblue')
    ax.text(0.0,
            max(train_label) + 0.3,
            'Epoch = %d' % e,
            fontdict={
                'size': 12,
                'color': 'red'
            })
    ax.text(0.0,
            max(train_label) + 0.15,
            'Loss = %.4f' % np.mean(losses),
            fontdict={
                'size': 12,
                'color': 'red'
            })

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    return image


def plot_hist(net):
    pred_hist = [
        net(Variable(torch.Tensor(np.matrix([0.0])))).data.numpy()[0][0]
        for _ in range(100)
    ]
    plt.hist(pred_hist, density=True, bins=30, label='x=0.0')

    pred_hist = [
        net(Variable(torch.Tensor(np.matrix([0.9])))).data.numpy()[0][0]
        for _ in range(100)
    ]
    # print(pred_hist)
    plt.hist(pred_hist, density=True, bins=30, label='x=0.9')
    plt.xlabel("Sampled y's")
    plt.ylabel("y values")
    plt.legend()
    plt.show()


def plot_loss(loss):
    plt.plot(loss)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_like(x, like):
    plt.plot(x, like, '-o', color='purple')
    # plt.scatter(x, like, marker='o')

    plt.xlabel("x")
    plt.ylabel("Log likelihood of true y under posterior distribution HUH")
    plt.show()


def plot_like_3d(x_val, like):
    # # print(x_val.shape)
    # x1, x2 = x_val.T[0], x_val.T[1]
    # # print(x1.shape, x2.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x_val[:, 0],
               x_val[:, 1],
               like,
               linewidth=0,
               c=like,
               cmap='coolwarm',
               alpha=0.9,
               marker='>',
               label='Log likelihood of true y under posterior distribution')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # ax.set_zlabel()

    plt.legend()
    plt.show()


def plot_like_contour(x_train, like, hp):
    fig, ax = plt.subplots()
    x1, x2 = hp.grid

    con = ax.contourf(x1, x2, np.array(like).reshape(x1.shape), cmap='plasma')

    # Sampled training data points
    ax.scatter(x_train[:, 0],
               x_train[:, 1],
               color='black',
               marker='+',
               alpha=.6,
               label='Train data')

    # Colour bar for uncertainty
    cbar = plt.colorbar(con)
    cbar.ax.set_ylabel('Log likelihood', fontsize=9)

    ax.set_aspect('equal')

    plt.legend()
    plt.show()
