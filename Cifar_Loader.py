import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    y = np.array(dict[b'labels'])
    return X, y


def load(directory, max_data_size, part_validation):
    path = directory + '/data_batch_1'  # todo: load all data
    X, y = unpickle(path)

    data_end_index = min(max_data_size, len(X))
    training_end_index = int(data_end_index * (1 - part_validation))
    Xtr = X[:training_end_index]
    ytr = y[:training_end_index]
    Xva = X[training_end_index + 1:]
    yva = y[training_end_index + 1:]
    return {'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva}
