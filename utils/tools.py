import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def transform_data(data):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaler, scaled_data


def inverse_data(scaler, data):
    ori_data = scaler.inverse_transform(data)
    return ori_data


def ood_data(X, sigma):
    return 