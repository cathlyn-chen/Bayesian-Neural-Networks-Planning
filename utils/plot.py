import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


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
                     pred_std):
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
    plt.show()


def plot_train_gif(fig, ax, train_data, train_label, predictions, losses,
                   x_test, y_true, e):
    plt.cla()
    ax.set_title('Training Progress', fontsize=21)
    ax.set_xlabel('Data', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    # ax.set_xlim(-0.9, 2.4)
    ax.set_ylim(min(train_label) - 0.3, max(train_label) + 0.45)
    ax.scatter(train_data, train_label, color="black")
    ax.plot(x_test, y_true, label='Truth', color='grey')
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
    plt.show()