import math
import numpy as np


def linear(x):
    """
    Linear activation function
    :param x:
    :return: x
    """
    return x


def sign(x):
    """
    Signum activation function
    :param x:
    :return: 1 for positive, -1 for negative numbers of X
    """
    return 1 if x > 0 else -1


def tanh(x):
    return math.tanh(x)


def hinge(y_true, y_phred):
    return max(1 - y_phred * y_true, 0)


def mean_squared_error(y_true, y_pred):
    """
    Mean squared error, returns the mean squared error of the prediction
    :param y_true:
    :param y_pred:
    :return:
    """
    return (y_pred - y_true) ** 2


def mean_absolute_error(y_true, y_pred):
    """
    returns the mean absolute error of the prediction
    :param y_true:
    :param y_pred:
    :return:
    """
    return abs(y_pred - y_true)


def derivative(function, delta=0.01):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + "'"
    wrapper_derivative.__qualname__ = function.__qualname__ + "'"
    return wrapper_derivative


class Perceptron:
    def __init__(self, dim, bias=0, weights=None, learning_rate=0.01):
        """

        :param dim:
        :param bias:
        :param weights:
        :param learning_rate:
        """
        self.dim = dim
        self.bias = bias
        self.learning_rate = learning_rate
        if weights is None:
            # bij geen opgegeven weights gelijk stellen aan 0 en de opgegeven dimensie
            self.weights = [0] * dim
        else:
            # aantal weights gelijk gesteld aan de dimensie
            assert len(weights) == dim, "Length of weights should match the dimensionality"
            self.weights = weights

    def predict(self, xs):
        """

        :param xs:
        :return:
        """
        predictions = []
        for instance in xs:
            prediction = self.predict_instance(instance)
            predictions.append(prediction)
        return predictions

    def predict_instance(self, instance):
        """
        :param instance:
        :return:
        """
        prediction = self.bias
        for i in range(len(instance)):
            prediction += self.weights[i] * instance[i]
        return numpy.sign(prediction)

    def partial_fit(self, xs, ys):
        """

        :param self:
        :param xs:
        :param ys:
        :return:
        """
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            self.bias -= (prediction - y)
            for i in range(len(x)):
                self.weights[i] -= (prediction - y) * x[i]

    def fit(self, xs, ys, *, epochs=0):
        """

        :param self:
        :param xs:
        :param ys:
        :param epochs:
        :return:
        """
        prev_weights = self.weights
        prev_bias = self.bias
        epoch = 0
        while True:
            self.partial_fit(xs, ys)
            if prev_weights == self.weights and prev_bias == self.bias:
                break  # geen veranderingen in de weights en bias dus stoppen
            # updaten van weight en bias
            prev_weights = self.weights.copy()
            prev_bias = self.bias
            epoch += 1
            if epochs != 0 and epoch >= epochs:
                break  # stop als het opgeven max aantal epochs behaald is

    def __repr__(self):
        text = f'Perceptron(dim={self.dim}, bias={self.bias}, weights={self.weights})'
        return text


class LinearRegression:
    def __init__(self, dim, bias=0, weights=None, learning_rate=0.01):
        self.dim = dim
        self.bias = bias
        self.learning_rate = learning_rate
        if weights is None:
            # bij geen opgegeven weights gelijk stellen aan 0 en de opgegeven dimensie
            self.weights = [0] * dim
        else:
            # aantal weights gelijk gesteld aan de dimensie
            assert len(weights) == dim, "Length of weights should match the dimensionality"
            self.weights = weights

    def predict(self, xs):
        predictions = []
        for instance in xs:
            prediction = self.predict_instance(instance)
            predictions.append(prediction)
        return predictions

    def predict_instance(self, instance):
        prediction = self.bias
        for i in range(len(instance)):
            prediction += self.weights[i] * instance[i]
        return prediction

    def partial_fit(self, xs, ys):
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            self.bias -= self.learning_rate * (prediction - y)
            for i in range(len(x)):
                self.weights[i] -= self.learning_rate * (prediction - y) * x[i]

    def fit(self, xs, ys, *, epochs=0):
        prev_weights = self.weights
        prev_bias = self.bias
        epoch = 0
        while True:
            self.partial_fit(xs, ys)
            if prev_weights == self.weights and prev_bias == self.bias:
                break  # geen veranderingen in de weights en bias dus stoppen
            # updaten van weight en bias
            prev_weights = self.weights.copy()
            prev_bias = self.bias
            epoch += 1
            if epochs != 0 and epoch >= epochs:
                break  # stop als het opgeven max aantal epochs behaald is

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim}, bias={self.bias}, weights={self.weights})'
        return text


class Neuron:
    def __init__(self, dim, activation=linear, loss=mean_squared_error):
        self.dim = dim
        self.bias = 0
        self.weights = [0] * dim
        self.activation = activation
        self.loss = loss

    def predict(self, xs):
        predictions = []
        for instance in xs:
            prediction = self.predict_instance(instance)
            predictions.append(prediction)
        return predictions

    def predict_instance(self, instance):
        prediction = self.bias
        for i in range(len(instance)):
            prediction += self.weights[i] * instance[i]
        return self.activation(prediction)

    def partial_fit(self, xs, ys, alpha=0.001):
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            loss_gradient = derivative(self.loss)(prediction, y)
            activation_gradient = derivative(self.activation)(self.bias + sum(w * xi for w, xi in zip(self.weights, x)))
            self.bias -= alpha * loss_gradient * activation_gradient
            for i in range(len(x)):
                self.weights[i] -= alpha * loss_gradient * activation_gradient * x[i]

    def fit(self, xs, ys, *, epochs=0):
        prev_weights = self.weights
        prev_bias = self.bias
        epoch = 0
        while True:
            self.partial_fit(xs, ys)
            if prev_weights == self.weights and prev_bias == self.bias:
                break  # geen veranderingen in de weights en bias dus stoppen
            # updaten van weight en bias
            prev_weights = self.weights.copy()
            prev_bias = self.bias
            epoch += 1
            if epochs != 0 and epoch >= epochs:
                break  # stop als het opgeven max. aantal epochs behaald is

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text
