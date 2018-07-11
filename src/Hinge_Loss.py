import numpy as np


class Hinge_Loss(object):
    # todo: support batches
    def __init__(self):
        self.delta = 1.0
        self.losses = None
        self.yi = None

    def forward(self, scores, yi):
        predicted_class = np.argmax(scores)
        self.yi = yi
        self.losses = np.zeros_like(scores)  # zero on each forward pass
        score_correct = scores[self.yi]
        for i in range(self.losses.size):
            self.losses[i] = max(0, scores[i] + self.delta - score_correct)
        self.losses[self.yi] = 0
        loss = np.sum(self.losses)
        return predicted_class, loss

    def backward(self, dl=1):  # just in case this is ever not last layer?
        ds = np.zeros_like(self.losses)
        for i, e in enumerate(self.losses):
            if e > 0:
                ds[i] = 1
        ds[self.yi] = -1 * sum(ds)  # number of classifiers with loss
        ds *= dl
        return ds

    def update(self, learning_rate):
        pass
