import numpy as np

num_batches = 5
examples_per_batch = 10000
num_examples = num_batches * examples_per_batch
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    y = np.array(dict[b'labels'])
    return X, y


def load(directory, max_data_size=num_examples, part_validation=0.1):

    X = np.zeros(shape=[num_examples, img_size_flat], dtype=np.float64)
    y = np.empty(shape=[num_examples], dtype=np.int64)

    for i in range(0, num_batches):
        path = directory + '/data_batch_%i' % (i+1)  # todo: load all data
        X_batch, y_batch = unpickle(path)
        X[i * examples_per_batch: (i + 1) * examples_per_batch] = X_batch
        y[i * examples_per_batch: (i + 1) * examples_per_batch] = y_batch

    data_end_index = min(max_data_size, len(X))
    training_end_index = int(data_end_index * (1 - part_validation))
    Xtr = X[:training_end_index]
    ytr = y[:training_end_index]
    Xva = X[training_end_index + 1:]
    yva = y[training_end_index + 1:]

    return {'Xtr': Xtr, 'ytr': ytr, 'Xva': Xva, 'yva': yva}
