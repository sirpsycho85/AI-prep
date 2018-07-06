import numpy as np
import Cifar_Loader as cifar


class Loss_Functions(object):
    def __init__(self):
        pass

    def hinge_loss(self, scores, index_correct):
        delta = 1.0
        score_correct = scores[index_correct]
        scores[index_correct] = 0
        loss = np.sum(scores) + delta - score_correct
        return loss


class Linear_Classifier(object):
    def __init__(self, loss_function, example_length, num_classes):
        self.example_length = example_length
        self.num_classes = num_classes
        self.W = np.random.random((num_classes, example_length))/1000
        self.loss_function = loss_function

    # todo: actual gradients based on backprop thru hinge loss and W
    def gradient_descent(self, loss, rate):
        gradients = np.zeros((num_classes, example_length))/1000
        return self.W + gradients * rate

    def train(self, X, y, rate):
        batch_scores = X.dot(self.W.T)
        loss = 0
        for example_scores, example_label in zip(batch_scores, y):
            loss += self.loss_function(example_scores, example_label)
        self.W = self.gradient_descent(loss, rate)

    def predict(self, X):
        scores = X.dot(self.W.T)
        return np.apply_along_axis(np.argmax, 1, scores)


data = cifar.load('cifar-10', max_data_size=1000, part_validation=0.1)
Xtr = data['Xtr']
ytr = data['ytr']
Xva = data['Xva']
yva = data['yva']
example_length = 32 * 32 * 3
num_classes = 10
lc = Linear_Classifier(Loss_Functions().hinge_loss, example_length, num_classes)
lc.train(Xtr, ytr, 0.001)

predicted_classes = lc.predict(Xva)
accuracy = np.mean(predicted_classes == yva)
print("Accuracy: %f" % accuracy)