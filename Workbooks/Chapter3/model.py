import math
import random
from collections import Counter
from copy import deepcopy


# activation functions
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
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return 0


def tanh(x):
    """
    Tanh activation function
    returns the tanh value of given  x
    :param x:
    :return:
    """
    return math.tanh(x)


# error/ loss functions
def hinge(y_true, y_phred):
    """
    Hinge error function
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


# derivative wrapper

def derivative(function, delta=0.001):
    """
    Calculates the derivative of a function for the given value.
    used for calculation of the slope at a given point.
    :param function:
    :param delta: the step size to use in derivative calculation
    :return:
    """

    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    # Give it a distinct name
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    # Return the wrapper function
    return wrapper_derivative


# perceptron, linearregression and neuron classes


class Perceptron:
    """
    A simple perceptron class
    used for solving binary linear separable data
    """

    def __init__(self, dim, bias=0, weights=None):
        """

        """
        self.dim = dim
        self.bias = bias
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
        :param xs: List of instances (data)
        :param ys: List of target / values
        :return:
        """
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            error = y - prediction
            self.bias += - error
            for i in range(len(x)):
                self.weights[i] += - error * x[i]

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

    def __init__(self, dim, bias=0, weights=None):
        self.dim = dim
        self.bias = bias

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
        :param xs: List of instances (data)
        :param ys: List of target / values
        :return:
        """
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            error = y - prediction
            self.bias += self.learning_rate * error
            for i in range(len(x)):
                self.weights[i] += self.learning_rate * error * x[i]

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

        return [self.activation(self.bias + sum(w * x for w, x in zip(self.weights, instance))) for instance in xs]

    def partial_fit(self, xs, ys, alpha=0.01):
        for x, y in zip(xs, ys):
            # calculate the prediction
            prediction = self.bias + sum(w * xi for w, xi in zip(self.weights, x))

            # calculate the  derivatives
            loss_gradient = derivative(self.loss)(prediction, y)
            activation_gradient = derivative(self.activation)(prediction)
            update_factor = alpha * loss_gradient * activation_gradient

            # Updating the bias
            self.bias -= update_factor

            # Updating weights
            self.weights = [w - update_factor * xi for w, xi in zip(self.weights, x)]

    def fit(self, xs, ys, epochs=0, alpha=0.001):
        prev_weights = self.weights.copy()
        prev_bias = self.bias
        epoch = 0
        while True:
            self.partial_fit(xs, ys, alpha=alpha)
            if prev_weights == self.weights and prev_bias == self.bias:
                print(f"Converged after {epoch} epochs.")
                break
            prev_weights = self.weights.copy()
            prev_bias = self.bias
            epoch += 1
            if epochs != 0 and epoch >= epochs:
                print(f"Stopped after reaching the max epochs: {epochs}.")
                break

    def __repr__(self):
        return f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'


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

    def __call__(self, xs):
        raise NotImplementedError('Abstract __call__ method')

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


class InputLayer(Layer):

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys=ys, alpha=alpha)

    def set_inputs(self, inputs):
        raise NotImplementedError("An InputLayer itself can not receive inputs from previous layers,"
                                  "as it is always the first layer of a network.")

    def predict(self, xs):
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        _, ls, _ = self(xs, ys)
        loss_mean = sum(ls) / len(ls)
        return loss_mean

    def partial_fit(self, xs, ys, alpha=0.001):
        self(xs, ys, alpha)

    def fit(self, xs, ys, epochs=800, alpha=0.001):
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


class DenseLayer(Layer):

    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)
        self.bias = [0.0 for _ in range(self.outputs)]
        self.weights = None

    def set_inputs(self, inputs):
        self.inputs = inputs

        if self.weights is None:
            # initialize weights using Xavier initialization for each neuron in the layer
            limit = math.sqrt(6 / (inputs + self.outputs))
            self.weights = [[random.uniform(-limit, limit) for _ in range(inputs)] for _ in range(self.outputs)]
        else:
            raise ValueError("Inputs already set.")

    def __call__(self, xs, ys=None, alpha=None):
        """

        :param xs:
        :param ys:
        :param alpha:
        :return:
        """
        activated_outputs = []  # Output values for all instances xs

        # forward propagation
        for x in xs:
            instance_output = [self.bias[o] + sum(wi * xi for wi, xi in zip(self.weights[o], x))
                               for o in range(self.outputs)]  # Output value for one instance x
            activated_outputs.append(instance_output)

        # check if training or not
        if ys is not None and alpha is not None:
            yhats, loss, gradients = self.next(activated_outputs, ys, alpha=alpha)
            gradient_x_list = []

            # backwards propogation updating  biases and weights
            for x, gradient_x in zip(xs, gradients):
                instance_gradient = [sum(self.weights[o][i] * gradient_x[o] for o in range(self.outputs)) for i in
                                     range(self.inputs)]
                gradient_x_list.append(instance_gradient)

                for o in range(self.outputs):
                    self.bias[o] -= alpha * gradient_x[o] / len(xs)
                    for i in range(self.inputs):
                        self.weights[o][i] -= alpha * gradient_x[o] * x[i] / len(xs)

            return yhats, loss, gradient_x_list
        else:
            # not training, return the forward pass results
            return activated_outputs, None, None


class ActivationLayer(Layer):

    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        self.activation = activation
        self.activation_derivative = derivative(self.activation)

    def __repr__(self):
        text = (f"{type(self).__name__}("f"num_outputs={self.outputs},"
                f" "f"name='{self.name}',"f" "f"activation='{self.activation.__name__}'"")")
        if self.next is not None:
            text += f" + {self.next!r}"
        return text

    def __call__(self, xs, ys=None, alpha=None):

        activated_outputs = []
        for x in xs:
            # calculate the activated output for each input x
            activated_output = [self.activation(x[o]) for o in range(self.outputs)]
            activated_outputs.append(activated_output)

        # if training check do backwards propogation
        if ys is not None and alpha is not None:
            yhats, loss, gradients = self.next(activated_outputs, ys, alpha=alpha)

            # calculate the gradients from the loss to the pre-activation value
            gradients_to_pre_activations = []
            for x, gradient in zip(xs, gradients):
                gradient_to_pre_activation = [self.activation_derivative(x[o]) * gradient[o] for o in
                                              range(self.outputs)]
                gradients_to_pre_activations.append(gradient_to_pre_activation)

            return yhats, loss, gradients_to_pre_activations
        else:
            return activated_outputs, None, None


class LossLayer(Layer):
    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(outputs=None, name=name)
        self.loss = loss

    def __repr__(self):
        text = f'LossLayer(loss={self.loss.__name__}, name={self.name})'
        return text

    def add(self, next):
        raise NotImplementedError("It is not possible to add a layer to a LossLayer,"
                                  "since a network should always end with a single LossLayer")

    def __call__(self, predictions, ys=None, alpha=None):

        yhats = predictions
        losses = None
        gradient_vector_list = None

        # calculate loss if targets are provided
        if ys:
            losses = [self.loss(yhat, y) for yhat, y in zip(yhats, ys)]

            # calculate gradients for training
            if alpha:
                gradient_vector_list = [
                    [derivative(self.loss)(yhat_i, y_i) for yhat_i, y_i in zip(yhat, y)]
                    for yhat, y in zip(yhats, ys)
                ]

        # outputs for further use
        return yhats, losses, gradient_vector_list
