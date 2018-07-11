import numpy as np
import Cifar_Loader as cifar
from Fully_Connected import Fully_Connected
from Hinge_Loss import Hinge_Loss


class Linear_Classifier(object):
    def __init__(self, loss_function, example_length, num_classes):
        self.example_length = example_length
        self.num_classes = num_classes
        self.W = np.random.random((num_classes, example_length))/1000
        self.loss_function = loss_function

    # todo: actual gradients based on backprop thru hinge loss and W
    # if you want to process a batch, you calculate the total loss for all samples
    # but do you sum gradients as well? SVM gradient function is different if it's the labeled class or not
    # also need to add biases, or use bias trick
    def update(self, losses, rate, y):
        gradients = np.zeros((num_classes, num_features))
        loss_gradients = self.loss_function.backward(losses, y)
        return self.W + gradients * rate

    # train on all samples
    def train(self, X, y, rate):
        batch_scores = X.dot(self.W.T)  # (num_samples, num_classes)
        losses = np.zeros(batch_scores.shape[1])  # (num_classes,)
        gradients = np.zeros_like(losses)  # (num_classes,)
        for example_scores, example_label in zip(batch_scores, y):
            example_loss = self.loss_function.forward(example_scores, example_label)
            losses += example_loss
            example_gradients = np.zeros((num_classes, num_features))

        self.W = self.update(losses, rate, y)

    def predict(self, X):
        scores = X.dot(self.W.T)
        return np.apply_along_axis(np.argmax, 1, scores)


class Neural_Network(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, X, y, learning_rate):
        pass


data = cifar.load('cifar-10', max_data_size=100, part_validation=0.1)
Xtr = data['Xtr']
Ytr = data['Ytr']
Xval = data['Xval']
Yval = data['Yval']
num_features = 32 * 32 * 3
num_classes = 10
# lc = Linear_Classifier(Hinge_Loss(), num_features, num_classes)
# lc.train(Xtr, Ytr, 0.001)
# predicted_classes = lc.predict(Xval)
# accuracy = np.mean(predicted_classes == Yval)
# print("Accuracy: %f" % accuracy)

fc = Fully_Connected(num_classes, num_features)
hinge = Hinge_Loss()
nn = Neural_Network()
nn.add_layer(fc)
nn.add_layer(hinge)
print(nn.layers)
