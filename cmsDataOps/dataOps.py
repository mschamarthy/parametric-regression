from math import floor
from numpy import zeros, ndarray, random


def data_split(data):
    row, col = data.shape
    X = zeros(shape=(row, col-1))
    X[:, :] = data[:, :col-1]
    y = zeros(shape=(row, 1))
    y[:, :] = data[:, col-1:]

    return X, y


def data_partition(x, n):
    row, col = x.shape
    split_size = row / n
    y = ndarray(shape=(n, split_size, col))
    for i in range(0, n):
        y[i, :, :] = x[i*split_size: ((i+1)*split_size), :]
    return y


def data_combine_except(x, i):
    splits, row, col = x.shape
    training = ndarray(shape=((splits-1)*row, col))
    testing = ndarray(shape=(row, col))
    testing = x[i, :, :]
    j = 0
    for k in range(0, splits):
        if k == i:
            continue
        training[j*row:(j+1)*row, :] = x[k, :, :]
        j += 1
    return training, testing


def data_sample(data, percent):
    rows = data.shape[0]
    if not (0 < percent <= 100):
        percent = 100
    rows_select = int(floor((percent/100.0) * rows))
    output = data[random.choice(rows, rows_select, replace=False), :]
    return output


def data_part_by_percent(data, percent, randomize=False):
    rows = data.shape[0]
    if not (0 < percent <= 100):
        percent = 100
    if randomize:
        random.shuffle(data)
    rows_select = int(floor((percent/100.0) * rows))
    selected_data = data[0:rows_select, :]
    remaining_data = data[rows_select:, :]
    return selected_data, remaining_data