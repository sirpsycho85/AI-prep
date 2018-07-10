import numpy as np


class Fully_Connected(object):  # todo: should inject W so you can customize initialization and just stack bias here?
    def __init__(self, num_classes, num_features):
        self.W = np.random.random((num_classes, num_features + 1)) / 1  # (num_classes, num_features + 1) bias trick
        self.x = None
        self.dw = None

    def forward(self, x):
        self.x = np.append(x, [1])  # (num_features + 1,)
        return self.W.dot(self.x)  # (num_classes,)

    def backward(self, ds):  # ds is dL/ds, where s is the scores output of the FC layer
        ds_T = np.reshape(ds, (-1, 1))  # ds to column vector so each row of ds gives row of dw for its classifier
        x = np.reshape(self.x, (1, -1))
        self.dw = ds_T.dot(x)
        return self.dw

    def update(self, learning_rate):
        self.W = self.W - learning_rate * self.dw
