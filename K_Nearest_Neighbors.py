import numpy as np
import Cifar_Loader as cifar


class K_Nearest_Neighbors(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1):
        prediction_batch_size = X.shape[0]
        ypreds = np.zeros(prediction_batch_size, dtype=self.ytr.dtype)
    
        for i in range(prediction_batch_size):

            distances = np.sum(np.square(self.Xtr - X[i, :]), axis=1)
            # X[i,:] means index i for 1st dimension, all elements of second dimension.
            # ...so ith image, all pixels

            k_nearest_indexes = np.argsort(distances)[0:k]
            classes_for_k_nearest = self.ytr[k_nearest_indexes]
            ypred = np.argmax(np.bincount(classes_for_k_nearest))
            ypreds[i] = ypred
        return ypreds


data = cifar.load('cifar-10', max_data_size=100, part_validation=0.1)
Xtr = data['Xtr']
ytr = data['ytr']
Xva = data['Xva']
yva = data['yva']

for k in [1, 3, 5, 10, 20, 50, 100]:
    nn = K_Nearest_Neighbors()
    nn.train(Xtr, ytr)
    yva_predict = nn.predict(Xva, k)
    acc = np.mean(yva_predict == yva)
    print('k: %i,\taccuracy: %f' % (k, acc))
