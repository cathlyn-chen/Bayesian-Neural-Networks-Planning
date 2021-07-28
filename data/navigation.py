import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(3)


def nav_sample(hp):
    epsilon_x = np.random.normal(0, hp.noise, size=(hp.train_size, ))

    epsilon_y = np.random.normal(0, hp.noise, size=(hp.train_size, ))

    # x1 = np.concatenate(
    #     (np.random.uniform(0.6, hp.bound, size=(hp.train_size // 3, )),
    #      np.tile(0.6, hp.train_size // 3),
    #      np.random.uniform(0.6, hp.bound, size=(hp.train_size // 3, ))),
    #     axis=0)
    # y1 = np.concatenate(
    #     (np.tile(5.1, hp.train_size // 3),
    #      np.random.uniform(0.0, hp.bound, size=(hp.train_size // 3, )),
    #      np.random.uniform(0.6, hp.bound, size=(hp.train_size // 3, ))),
    #     axis=0)

    # x1 = np.concatenate(
    #     (np.random.uniform(0.9, hp.bound, size=(hp.train_size // 2, )),
    #      np.tile(0.6, hp.train_size // 2)),
    #     axis=0)
    # y1 = np.concatenate(
    #     (np.tile(5.1, hp.train_size // 2),
    #      np.random.uniform(0.0, hp.bound, size=(hp.train_size // 2, ))),
    #     axis=0)

    # x1 = np.concatenate(
    #     (np.random.uniform(0.9, hp.bound, size=(hp.train_size // 2, )),
    #      np.random.uniform(0.9, hp.bound, size=(hp.train_size // 2, ))),
    #     axis=0)
    # y1 = np.concatenate(
    #     (np.tile(5.1, hp.train_size // 2), np.tile(0.6, hp.train_size // 2)),
    #     axis=0)

    # x1 = np.random.uniform(0.6, hp.bound-0.6, size=(hp.train_size))
    # y1 = np.tile(0.9, hp.train_size)

    x1 = np.concatenate(
        (np.random.uniform(0.6, hp.bound - 0.6, size=(hp.train_size // 2, )),
         np.random.uniform(0.6, hp.bound, size=(hp.train_size // 2, ))),
        axis=0)
    y1 = np.concatenate(
        (np.tile(5.1, hp.train_size // 2),
         np.random.uniform(0.6, hp.bound, size=(hp.train_size // 2, ))),
        axis=0)

    # x2 = x1 + epsilon_x
    # y2 = y1 + epsilon_y

    # x2 = np.random.uniform(0.0, hp.bound, size=x1.shape)
    # y2 = np.random.uniform(0.0, hp.bound, size=y1.shape)

    # x2 = np.concatenate(
    #     (np.random.uniform(0.9, hp.bound, size=(hp.train_size // 3, )),
    #      np.tile(0.6, hp.train_size // 3),
    #      np.random.uniform(0.9, hp.bound, size=(hp.train_size // 3, ))),
    #     axis=0)
    # y2 = np.concatenate(
    #     (np.tile(5.1, hp.train_size // 3),
    #      np.random.uniform(0.0, hp.bound, size=(hp.train_size // 3, )),
    #      np.random.uniform(0.6, hp.bound, size=(hp.train_size // 3, ))),
    #     axis=0)
    # x1 = x2 + epsilon_x
    # y1 = y2 + epsilon_y

    return x1, epsilon_x, y1, epsilon_y
    # return x1, x2, y1, y2


def nav_data(x1, x2, y1, y2, hp):
    # x_delta = x2 - x1
    # y_delta = y2 - y1

    # 60 x 4
    train_data = np.dstack((x1, x2, y1, y2)).reshape(hp.train_size, -1)
    # 60 x 2
    # train_label = np.dstack((x_delta, y_delta)).reshape(hp.train_size, -1)
    train_label = np.dstack((x2, y2)).reshape(hp.train_size, -1)

    # 100 x 2
    test_x1_y1 = hp.grid.copy().reshape(2, -1)
    # 100 x 2
    test_x2_y2 = np.random.uniform(0.0, hp.noise, size=test_x1_y1.shape)
    # 100 x 4
    test_data = np.dstack((test_x1_y1, test_x2_y2)).reshape(-1, 4)
    # 100 x 2
    # test_label = np.dstack((test_x2_y2[0] - test_x1_y1[0],
    #                         test_x2_y2[1] - test_x1_y1[1])).reshape(-1, 2)
    test_label = np.dstack((test_x2_y2[0], test_x2_y2[1])).reshape(-1, 2)

    return train_data, train_label, test_data, test_label


def nav_data_pair(x1, x2, y1, y2, hp):
    x_delta = x2 - x1
    y_delta = y2 - y1

    # 60 x 4
    train_data_x = np.dstack((x1, x2)).reshape(hp.train_size, -1)
    train_data_y = np.dstack((y1, y2)).reshape(hp.train_size, -1)

    test_x1_y1 = hp.grid.copy().reshape(2, -1)
    test_x2_y2 = np.random.uniform(0.0, hp.bound, size=test_x1_y1.shape)

    test_data_x = np.dstack((test_x1_y1[0], test_x2_y2[0])).reshape(-1, 2)
    test_data_y = np.dstack((test_x1_y1[1], test_x2_y2[1])).reshape(-1, 2)
    test_label_x = test_x2_y2[0] - test_x1_y1[0]
    test_label_y = test_x2_y2[1] - test_x1_y1[1]

    return train_data_x, x_delta, train_data_y, y_delta, test_data_x, test_data_y, test_label_x, test_label_y
