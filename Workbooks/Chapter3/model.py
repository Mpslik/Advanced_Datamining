import math
import pandas as pd
import tensorflow as tf
from collections import Counter
from copy import deepcopy


def linear(x):
    """
    Linear activation function
    returns x
    :param x:
    :return: x
    """
    return x


def sign(x):
    """
    Signum activation function
    that return a value of -1 or 1 depending on the value of x
    -1 for negative numbers and 1 for positive numbers
    :param x:
    :return: 1 for positive, -1 for negative numbers of X
    """
    return 1 if x > 0 else -1


def tanh(x):
    """
    Tanh activation function
    returns the tanh value of given  x
    :param x:
    :return:
    """
    return math.tanh(x)


def hinge(y_true, y_phred):
    """
    Hinge activation function
    r
    :param y_true:
    :param y_phred:
    :return:
    """
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
    """
    Calculates the derivative of a function for the given value.
    used for calculation of the slope at a given point.
    :param function:
    :param delta: the step size to use in derivative calculation
    :return:
    """

    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + "'"
    wrapper_derivative.__qualname__ = function.__qualname__ + "'"
    return wrapper_derivative


class Perceptron:
    """
    A simple perceptron class
    used for solving binary linear separable data
    """

    def __init__(self, dim, bias=0, weights=None, learning_rate=0.01):
        """

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
        Calculates the prediction of the given list of instance with the stored weights and bias
        :param xs: a list of instances
        :return: list of predictions
        """
        predictions = []
        for instance in xs:
            prediction = self.predict_instance(instance)
            predictions.append(prediction)
        return predictions

    def predict_instance(self, instance):

        """
        predicts the given instance with the stored weights and bias
        :param instance: a single instance array
        :return: the predicted value int
        """
        prediction = self.bias
        for i in range(len(instance)):
            prediction += self.weights[i] * instance[i]
        return sign(prediction)

    def partial_fit(self, xs, ys):
        """
        Runs a partial fit of the model on the data
        1 single run
        :param self:
        :param xs: List of instances (data)
        :param ys: List of target / values
        :return:
        """
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            self.bias -= (prediction - y)
            for i in range(len(x)):
                self.weights[i] -= (prediction - y) * x[i]

    def fit(self, xs, ys, *, epochs=0):
        """
        update/fit the model on the data with either the given amount op epochs,
        or until there are no more changes in weight and bias.
        :param self:
        :param xs: List of instances (data)
        :param ys: List of target / values
        :param epochs: Int number of runs
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
    """
    linear regression model same as perceptron class but with a different activation.
    """

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
        """
        Calculates the prediction of the given list of instance with the stored weights and bias
        :param xs: a list of instances
        :return: list of predictions
        """
        predictions = []
        for instance in xs:
            prediction = self.predict_instance(instance)
            predictions.append(prediction)
        return predictions

    def predict_instance(self, instance):

        """
        predicts the given instance with the stored weights and bias
        :param instance: a single instance array
        :return: the predicted value int
        """
        prediction = self.bias
        for i in range(len(instance)):
            prediction += self.weights[i] * instance[i]
        return sign(prediction)

    def partial_fit(self, xs, ys):
        """
                Runs a partial fit of the model on the data
                1 single run
                :param self:
                :param xs: List of instances (data)
                :param ys: List of target / values
                :return:
                """
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
            prediction = self.bias
            for i in range(len(instance)):
                prediction += self.weights[i] * instance[i]
            predictions.append(self.activation(prediction))
        return predictions

    def partial_fit(self, xs, ys, alpha=0.001):
        for x, y in zip(xs, ys):
            prediction = self.bias
            for i in range(len(x)):
                prediction += self.weights[i] * x[i]

            loss_gradient = derivative(self.loss)(prediction, y)
            activation_gradient = derivative(self.activation)(prediction)

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
                break  # Stop als het opgeven max. aantal epochs behaald is

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text


class Layer:
    layercounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.layercounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.layercounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs

    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')

    def __iadd__(self, next):
        self.__add__(next)
        return self

    def __len__(self):
        return len(self)

    def __iter__(self):
        return iter(self)


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


class DenseLayer(Layer):

    def __init__(self, outputs, *, name=None, next=None):
        super().__init__()
        self.name = name
        self.output = outputs
        self.next = next




    def set_inputs(self, inputs):
        if self.weights is None:
            # Initialize weights using Xavier initialization
            limit = math.sqrt(6 / (inputs + 1))
            self.weights = [random.uniform(-limit, limit) for _ in range(inputs)]
        else:
            raise ValueError("Inputs already set.")
