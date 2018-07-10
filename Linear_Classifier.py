import numpy as np
import Cifar_Loader as cifar


"""
Training:

For each example x
    Forward through FC:
        Add bias trick "feature" = 1
        Multiply W (num_classes, num_features+1) by x (num_features,) to produce scores (num_classes,)
    Forward through hinge loss:
        Calculate class_losses (num_classes,) as sum(max(0, score[class_index] + delta - score[index_correct_class]))
        Zero out loss[index_correct_class]
        Calculate loss (1,) as sum(class_losses)
    Backwards through hinge loss to calculate d_class_score (num_classes,):
        For each classifier with loss > 0, d_class_score = 1
        For the classifier of the correct class, d_class_score = -1 * num_classifiers_with_nonzero_loss
    Backwards through FC:
        Gradients (num_classes, num_features+1) = d_class_score * x
        Update FC: W = W - learning_rate * gradients
        
So FC has to remember X, hinge has to remember the individual classifier losses. These are the "local" gradients.       
"""

class Fully_Connected(object):
    def __init__(self, num_classes, num_features):
        self.W = np.random.random((num_classes, num_features + 1)) / 1000  # (num_classes, num_features + 1) bias trick
        self.x = None

    def forward(self, x):
        self.x = np.append(x, [1]) # (num_features + 1,)
        return self.W.dot(self.x)  # (num_classes,)

    def backward(self, dsdw): #ds := dL/ds, where s is the scores output of the FC layer
        dsdw_T = np.reshape(dsdw, (-1, 1))
        x = np.reshape(self.x, (1, -1))
        print(dsdw_T.shape)
        dw = dsdw_T.dot(x)
        return dw

    def update(self, learning_rate):
        pass

fc = Fully_Connected(3, 2)
fc_out = fc.forward(np.random.random(2))
print("fc_out = \n", fc_out, "\n")
fc_grad = fc.backward(np.ones(3))
print("fc_grad = \n", fc_grad, "\n")

class Hinge_Loss(object):
    def __init__(self):
        pass

    def forward(self, scores, index_correct):
        delta = 1.0
        score_correct = scores[index_correct]
        losses = np.zeros_like(scores)
        for i in range(losses.size):
            losses[i] = max(0, scores[i] + delta - score_correct)
        losses[index_correct] = 0
        # loss = np.sum(losses)
        return losses

    def backward(self, losses, index_correct):
        gradients = np.zeros_like(losses)
        for i, e in enumerate(losses):
            if e > 0:
                gradients[i] = 1
        gradients[index_correct] = -1
        return gradients


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
        gradients = np.zeros((num_classes, example_length))
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
            example_gradients = np.zeros((num_classes, example_length))

        self.W = self.update(losses, rate, y)

    def predict(self, X):
        scores = X.dot(self.W.T)
        return np.apply_along_axis(np.argmax, 1, scores)


data = cifar.load('cifar-10', max_data_size=100, part_validation=0.1)
Xtr = data['Xtr']
ytr = data['ytr']
Xval = data['Xval']
yval = data['yval']
example_length = 32 * 32 * 3
num_classes = 10
lc = Linear_Classifier(Hinge_Loss(), example_length, num_classes)
lc.train(Xtr, ytr, 0.001)

predicted_classes = lc.predict(Xval)
accuracy = np.mean(predicted_classes == yval)
# print("Accuracy: %f" % accuracy)