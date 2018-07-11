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
        self.loss_type = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

    def train(self, X, Y, learning_rate):
        batch_size = Y.size
        accurate_predictions = 0
        for i in range(batch_size):
            next_layer_input = X[i]
            for layer in self.layers:
                next_layer_input = layer.forward(next_layer_input)
            predicted_class, loss = self.loss_type.forward(next_layer_input, Y[i])
            print(predicted_class)
            accurate_predictions += 1
            # print(loss)
            next_layer_dL = self.loss_type.backward()
            for layer in reversed(self.layers):
                next_layer_dL = layer.backward(next_layer_dL)
                layer.update(learning_rate)
        # print("accurate predictions = ", accurate_predictions, "batch size = ", batch_size)


def normalize_and_zero_mean(arr, max=None):
    if max == None:
        max = np.max(arr)
    return arr/max - 1/2


# data = cifar.load('cifar-10', max_data_size=5, part_validation=0.2)
# Xtr = normalize_and_zero_mean(data['Xtr'])
# Ytr = data['Ytr']
# Xval = normalize_and_zero_mean(data['Xval'])
# Yval = data['Yval']
# num_features = 32 * 32 * 3
# num_classes = 10

# TRIVIAL EXAMPLE OF TWO CLASSES THAT ARE LINEARLY SEPARABLE TO DEBUG
Xtr = np.array([[1, 1], [0, 0], [0, 0], [0, 0]])
Ytr = np.array([1, 0, 0, 0])
num_classes = 2
num_features = 2

fc = Fully_Connected(num_classes, num_features)
hinge = Hinge_Loss()
nn = Neural_Network()
nn.add_layer(fc)
nn.set_loss_type(hinge)

num_epochs = 1000
for i in range(num_epochs):
    nn.train(Xtr, Ytr, 0.01)

# lc = Linear_Classifier(Hinge_Loss(), num_features, num_classes)
# lc.train(Xtr, Ytr, 0.001)
# predicted_classes = lc.predict(Xval)
# accuracy = np.mean(predicted_classes == Yval)
# print("Accuracy: %f" % accuracy)
