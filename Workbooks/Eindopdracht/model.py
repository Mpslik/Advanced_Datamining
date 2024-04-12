import math
import random
import time
from collections import Counter
from copy import deepcopy
from typing import List, Tuple, Callable, Any, Optional


# Activation functions
def linear(x: float) -> float:
    """
    Linear activation function
    :param x: input float
    :return: x
    """
    return x


def sign(x: float) -> float:
    """
    Signum activation function
    that returns a value of -1, 0, or 1 depending on the value of x
    :param x: input float
    :return: 1 for positive, -1 for negative, 0 for zero
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def tanh(x: float) -> float:
    """
    Hyperbolic tangent activation function
    :param x: input float
    :return: tanh(x)
    """
    return math.tanh(x)


def softsign(x: float) -> float:
    """
    Softsign activation function
    :param x: input float
    :return: the softsign of x
    """
    return x / (abs(x) + 1)


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function
    :param x: input float
    :return: the sigmoid value of x
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def softplus(x: float) -> float:
    """
    Softplus activation function
    :param x: input float
    :return: the softplus of x
    """
    return math.log(1 + math.exp(-abs(x))) + max(x, 0)


def relu(x: float) -> float:
    """
    ReLU activation function
    :param x: input float
    :return: max(0, x)
    """
    return max(0, x)


def swish(x: float) -> float:
    """
    Swish activation function
    :param x: input float
    :return: x multiplied by the sigmoid of x
    """
    return x * sigmoid(x)


def softmax(x: List[float]) -> List[float]:
    """
    Softmax activation function
    :param x: list of float
    :return: list of softmax probabilities
    """
    max_value = max(x)
    x_normalized = [value - max_value for value in x]
    sum_exp = sum(math.exp(xi) for xi in x_normalized)
    return [math.exp(xi) / sum_exp for xi in x_normalized]


def nipuna(x: float, beta: float = 1.0) -> float:
    """
    NIPUNA Activation Function to handle large inputs without overflow.
    :param x: input float
    :param beta: scaling parameter
    :return: output of the NIPUNA function
    """
    if -beta * x > 700:
        exp_term = 0.0
    else:
        exp_term = math.exp(-beta * x)
    gx = x / (1 + exp_term)
    return max(gx, x)


# error/ loss functions
def hinge(y_pred: float, y_true: float) -> float:
    """
    Hinge error function
    :param y_true:
    :param y_pred:
    :return:
    """
    return max(1 - y_pred * y_true, 0)


def mean_squared_error(y_pred: float, y_true: float) -> float:
    """
    Mean squared error, returns the mean squared error of the prediction
    :param y_true:
    :param y_pred:
    :return:
    """
    return (y_pred - y_true) ** 2


def mean_absolute_error(y_pred: float, y_true: float) -> float:
    """
    returns the mean absolute error of the prediction
    :param y_true:
    :param y_pred:
    :return:
    """
    return abs(y_pred - y_true)


def binary_crossentropy(y_pred: float, y_true: float, epsilon: float = 0.0001) -> float:
    """
    calculates the binary cross entropy loss.
    :param y_pred:
    :param y_true:
    :param epsilon:
    :return:
    """
    return -y_true * pseudo_log(y_pred, epsilon) - (1 - y_true) * pseudo_log(1 - y_pred, epsilon)


def categorical_crossentropy(y_pred: float, y_true: float, epsilon: float = 0.0001) -> float:
    """
    calculates the categorical cross entropy loss.
    :param y_pred:
    :param y_true:
    :param epsilon:
    :return:
    """
    return -y_true * pseudo_log(y_pred, epsilon)


def pseudo_log(x: float, epsilon: float = 0.001):
    """
    Provides a numerically stable logarithm calculation to prevent math domain errors.
    :param x:
    :param epsilon:
    :return:
    """
    if x < epsilon:
        return math.log(epsilon) + (x - epsilon) / epsilon
    return math.log(x)


# derivative wrapper

def derivative(function: Callable[..., float], delta: float = 0.001) -> Callable[..., float]:
    """
    Calculates the derivative of a function for the given value.
    Used for calculation of the slope at a given point.

    :param function: The function to differentiate, which should take one or more float arguments and return a float.
    :param delta: The step size to use in derivative calculation.
    :return: A function that calculates the derivative of the given function at a point.
    """

    def wrapper_derivative(x: float, *args: Any) -> float:
        """
        Wrapper function that computes the numerical derivative.

        :param x: The point at which to calculate the derivative.
        :param args: Additional arguments passed to the function.
        :return: The derivative of the function at point x.
        """
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    # Give it a distinct name
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    # Return the wrapper function
    return wrapper_derivative


class ProgressBar:
    def __init__(self, total_epochs: int, bar_width: int = 40) -> None:
        """
        Initialize the ProgressBar with the total number of epochs and optional bar width.

        :param total_epochs: Total number of epochs the progress bar will represent.
        :param bar_width: Optional width of the progress bar in characters. Default is 40.
        """
        self.total_epochs = total_epochs
        self.bar_width = bar_width
        self.last_epoch_time = time.time()

    def format_time(self, seconds: float) -> str:
        """
        Formats seconds into a string HH:MM:SS.mmm, including milliseconds for durations less than a second.

        :param seconds: Time duration in seconds to format.
        :return: Formatted time as a string.
        """
        # Dividing everything in the correct time format
        hours, remainder = divmod(int(seconds), 3600)
        minutes, int_seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int_seconds) * 1000)
        # Setting the returned string
        formatted_time = f"{hours:02d}:{minutes:02d}:{int_seconds:02d}.{milliseconds:03d}"
        return formatted_time

    def update(self, epoch: int, start_time: float) -> None:
        """
        Updates the progress bar display for the given epoch.

        :param epoch: The current epoch number.
        :param start_time: The start time of the process (for overall elapsed time calculation).
        """
        current_time = time.time()
        elapsed_time = current_time - start_time
        elapsed_str = self.format_time(elapsed_time)

        epoch_elapsed_time = current_time - self.last_epoch_time
        epoch_elapsed_str = self.format_time(epoch_elapsed_time)

        percent_complete = (epoch + 1) / self.total_epochs * 100
        completed_width = int(self.bar_width * percent_complete / 100)
        remaining_width = self.bar_width - completed_width
        progress_str = "▓" * completed_width + "░" * remaining_width

        epoch_info = f"\033[92mEpoch {epoch + 1}/{self.total_epochs}\033[0m"
        bar = f"[\033[94m{progress_str}\033[0m]"
        percent = f"\033[93m{percent_complete:.0f}%\033[0m"
        elapsed = f"\033[95mElapsed time: {elapsed_str}\033[0m"
        since_last = f"\033[96mTime since last epoch: {epoch_elapsed_str}\033[0m"

        print(f"\r{epoch_info} {bar} {percent} | {elapsed} | {since_last}", end="", flush=True)

        if epoch + 1 == self.total_epochs:
            print()  # Move to new line after final epoch

        self.last_epoch_time = current_time  # Update the last epoch time for next calculation


# perceptron, linearregression and neuron classes


class Perceptron:
    def __init__(self, dim: int, bias: float = 0, weights: Optional[List[float]] = None, learning_rate: float = 0.01):
        """
        Initialize the Perceptron with a specified dimension, bias, initial weights, and learning rate.

        :param dim: Dimensionality of the input features.
        :param bias: Initial bias value.
        :param weights: Initial list of weights; if None, weights are initialized to zero.
        :param learning_rate: The learning rate used in training.
        """
        self.dim = dim
        self.bias = bias
        self.learning_rate = learning_rate
        if weights is None:
            self.weights = [0.0] * dim
        else:
            assert len(weights) == dim, "Length of weights should match the dimensionality"
            self.weights = weights

    def predict(self, xs: List[List[float]]) -> List[int]:
        """
        Predict the label of each instance in the dataset.

        :param xs: A list of instances, where each instance is a list of feature values.
        :return: A list of predicted labels.
        """
        predictions = [self.predict_instance(instance) for instance in xs]
        return predictions

    def predict_instance(self, instance: List[float]) -> float:
        """
        Make a prediction for a single instance.

        :param instance: A single instance represented as a list of feature values.
        :return: The predicted label for the instance.
        """
        prediction = self.bias
        for weight, feature in zip(self.weights, instance):
            prediction += weight * feature
        return sign(prediction)

    def partial_fit(self, xs: List[List[float]], ys: List[float]):
        """
        Perform a partial fit for one epoch over the dataset.

        :param xs: A list of instances.
        :param ys: The true labels corresponding to each instance.
        """
        for x, y in zip(xs, ys):
            prediction = self.predict_instance(x)
            error = y - prediction
            self.bias += self.learning_rate * error
            self.weights = [w + self.learning_rate * error * xi for w, xi in zip(self.weights, x)]

    def fit(self, xs: List[List[float]], ys: List[float], *, epochs: int = 500) -> None:
        """
        Fit the Perceptron model to the data, stopping early if weights and bias do not change.

        :param xs: A list of instances.
        :param ys: A list of true labels for the instances.
        :param epochs: Maximum number of epochs for training.
        """
        prev_weights = self.weights.copy()
        prev_bias = self.bias
        for epoch in range(epochs):
            self.partial_fit(xs, ys)
            if prev_weights == self.weights and prev_bias == self.bias:
                print(f"Convergence reached after {epoch} epochs.")
                break  # Stop training if no changes
            prev_weights = self.weights.copy()
            prev_bias = self.bias

    def __repr__(self) -> str:
        return f'Perceptron(dim={self.dim}, bias={self.bias}, weights={self.weights})'


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

    def fit(self, xs, ys, epochs=500, alpha=0.001):
        prev_weights = self.weights.copy()
        prev_bias = self.bias
        epoch = 0

        print("fitting")
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

    def partial_fit(self, xs, ys, *, alpha=0.001, batch_size=None):
        """

        :param xs:
        :param ys:
        :param alpha:
        :param batch_size:
        :return:
        """
        input_length = len(xs)
        # determine the actual batch size
        batch_size = input_length if batch_size is None or batch_size >= input_length else batch_size

        total_loss = 0

        for batch_start in range(0, input_length, batch_size):
            batch_end = min(batch_start + batch_size, input_length)
            x_batch, y_batch = xs[batch_start:batch_end], ys[batch_start:batch_end]

            # assuming self(x_batch, y_batch, alpha) returns model predictions, batch loss, and updates model
            _, batch_losses, _ = self(x_batch, y_batch, alpha)
            total_loss += sum(batch_losses)

        mean_loss = total_loss / input_length  # calculate mean loss across all samples
        return mean_loss

    def fit(self, xs, ys, *, epochs=500, alpha=0.001, batch_size=None, validation_data=None):
        """
        :param xs:
        :param ys:
        :param epochs:
        :param alpha:
        :param batch_size:
        :param validation_data:
        :return:
        """
        start_time = time.time()
        history = {'loss': []}
        evaluate_validation = validation_data is not None
        if evaluate_validation:
            history['val_loss'] = []
            val_xs, val_ys = validation_data

        # Instantiate the ProgressBar
        progress_bar = ProgressBar(total_epochs=epochs)

        for epoch in range(epochs):
            # Updating the bar
            progress_bar.update(epoch=epoch, start_time=start_time)

            # combine xs and ys into a list of tuples and shuffle
            combined = list(zip(xs, ys))
            random.shuffle(combined)
            # unzip the shuffled list of tuples back into xs and ys
            xs_shuffled, ys_shuffled = zip(*combined)

            # train on the shuffled data and record mean loss for this epoch
            epoch_loss = self.partial_fit(xs_shuffled, ys_shuffled, alpha=alpha, batch_size=batch_size)
            history['loss'].append(epoch_loss)

            # evaluate on validation data and record loss
            if evaluate_validation:
                validation_loss = self.evaluate(val_xs, val_ys)
                history['val_loss'].append(validation_loss)

        return history

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
        yhats, loss, gradients = self.next(activated_outputs, ys, alpha=alpha)
        if ys is not None and alpha is not None:

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
            return yhats, loss, None


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

        yhats, loss, gradients = self.next(activated_outputs, ys, alpha=alpha)
        # if training check do backwards propogation
        if ys is not None and alpha is not None:

            # calculate the gradients from the loss to the pre-activation value
            gradients_to_pre_activations = []
            for x, gradient in zip(xs, gradients):
                gradient_to_pre_activation = [self.activation_derivative(x[o]) * gradient[o] for o in
                                              range(self.outputs)]
                gradients_to_pre_activations.append(gradient_to_pre_activation)

            return yhats, loss, gradients_to_pre_activations
        else:
            return yhats, loss, None


class LossLayer(Layer):
    def __init__(self, loss_func=mean_squared_error, name=None):
        super().__init__(outputs=None, name=name)
        self.loss_func = loss_func

    def __repr__(self):
        text = f'LossLayer(loss={self.loss_func.__name__}, name={self.name})'
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
            losses = []
            for yhat, y in zip(yhats, ys):
                # Take sum of the loss of all outputs(number of outputs previous layer=inputs this layer)
                loss = sum(self.loss_func(yhat[o], y[o]) for o in range(self.inputs))
                losses.append(loss)
            # calculate gradients for training
            if alpha:
                gradient_vector_list = [
                    [derivative(self.loss_func)(yhat_i, y_i) for yhat_i, y_i in zip(yhat, y)]
                    for yhat, y in zip(yhats, ys)
                ]

        # outputs for further use
        return yhats, losses, gradient_vector_list


class SoftmaxLayer(Layer):
    """

    """

    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)

    def __repr__(self):
        return f'SoftmaxLayer(outputs={self.outputs}, name={self.name})'

    def __call__(self, xs, ys=None, alpha=None):
        yhats = []  # Probability distributions for each instance
        gradients_to_h = None

        for x in xs:
            prob = softmax(x)
            yhats.append(prob)

        predictions, losses, gradients_from_loss = self.next(yhats, ys, alpha)

        if alpha and gradients_from_loss is not None:
            gradients_to_h = [
                [sum(gradients_from_loss[o] * prediction[o] * ((i == o) - prediction[i]) for o in range(self.outputs))
                 for i in range(self.inputs)] for prediction, gradients_from_loss in
                zip(predictions, gradients_from_loss)
            ]

        return predictions, losses, gradients_to_h
