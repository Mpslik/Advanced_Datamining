import math
import random
import time
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
    signum activation function
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
    return math.tanh(x)


def softsign(x):
    """
    Softsign activation function
    :param x:
    :return:
    """
    return x / (abs(x) + 1)


def sigmoid(x):
    """

    :param x:
    :return:
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        # For x < 0, avoid overflow with rearrangement
        z = math.exp(x)
        return z / (1 + z)


def softplus(x):
    """

    :param x:
    :return:
    """
    return math.log(1 + math.exp(-abs(x))) + max(x, 0)


def relu(x):
    """

    :param x:
    :return:
    """

    return max(0, x)


def swish(x):
    """

    :param x:
    :return:
    """
    return x * sigmoid(x)


def softmax(x):
    """

    :param x:
    :return:
    """
    # Normalize x to prevent OverflowError, by subtracting the max from the vector
    max_value = max(x)
    x_normalized = [value - max_value for value in x]
    # Calculate the sum of e^xi for every value xi in the normalized list x,
    # used as the denominator in the softmax function
    sum_exp = sum(math.exp(xi) for xi in x_normalized)
    # Apply softmax function to the normalized list x
    softmax_values = [math.exp(xi) / sum_exp for xi in x_normalized]
    # Return probability distribution
    return softmax_values


def nipuna(x, beta=1):
    """
    NIPUNA
    :param x:
    :param beta:
    :return: t
    """
    # Compute the numerator in the expression for g(x)
    if beta * x >= 0:
        g_x = x - math.log1p(math.exp(-beta * x))
    else:
        g_x = math.exp(beta * x) * x / (math.exp(beta * x) + 1) - math.log1p(math.exp(beta * x))

    # Compute the maximum of g(x) and x
    max_g_x = max(g_x, x)
    return max_g_x

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


def binary_crossentropy(y_pred, y_true, epsilon=0.0001):
    """
    calculates the binary cross entropy loss.
    :param y_pred:
    :param y_true:
    :param epsilon:
    :return:
    """
    return -y_true * pseudo_log(y_pred, epsilon) - (1 - y_true) * pseudo_log(1 - y_pred, epsilon)


def categorical_crossentropy(y_pred, y_true, epsilon=0.0001):
    """
    calculates the categorical cross entropy loss.
    :param y_pred:
    :param y_true:
    :param epsilon:
    :return:
    """
    return -y_true * pseudo_log(y_pred, epsilon)


def pseudo_log(x, epsilon=0.001):
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


def progress_bar(epoch, total_epochs, start_time, bar_width = 40):
    """
    Prints a progress bar to the console.
    :param epoch: the current epoch
    :param total_epochs: the total number of epochs
    :param start_time: the time when the training started
    :param bar_width: the width of the progress bar
    """
    # Get the current time and calculate the elapsed time
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Calculate the elapsed time in minutes and seconds
    elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)

    # Format the elapsed time as a string
    elapsed_str = f"{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}"

    # If it's not the first epoch
    if epoch > 0:
        # Get the time of the last epoch and calculate the elapsed time since then
        last_epoch_time = getattr(progress_bar, "last_epoch_time", current_time)
        epoch_elapsed_time = current_time - last_epoch_time

        # Calculate the elapsed time in minutes and seconds
        epoch_elapsed_minutes, epoch_elapsed_seconds = divmod(epoch_elapsed_time, 60)

        # Format the elapsed time since the last epoch as a string
        epoch_elapsed_str = f"{int(epoch_elapsed_minutes):02d}:{int(epoch_elapsed_seconds):02d}"
    else:
        # If it's the first epoch, set the elapsed time to 00:00
        epoch_elapsed_str = "00:00"

    # Save the current time as the time of the last epoch
    progress_bar.last_epoch_time = current_time

    # Calculate the percentage of epochs completed
    percent_complete = epoch / total_epochs * 100

    # Calculate the width of the progress bar
    completed_width = int(bar_width * percent_complete / 100)
    remaining_width = bar_width - completed_width

    # Create the progress bar
    completed_bar = "#" * completed_width
    remaining_bar = " " * remaining_width

    # Format the percentage as a string
    percentage_str = f"{percent_complete:.0f}%"

    print(
        f"\rEpoch {epoch + 1}/{total_epochs} [{completed_bar}{remaining_bar}] {percentage_str} | Elapsed time: {elapsed_str} | Time since last epoch: {epoch_elapsed_str}",
        end="",
        flush=True,
    )

# perceptron, linearregression and neuron classes


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
        return sign(prediction)

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

    def fit(self, xs, ys, *, epochs=500):
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

    def fit(self, xs, ys, *, epochs=500 , alpha=0.001, batch_size=None, validation_data=None):
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

        for epoch in range(epochs):
            progress_bar(epoch=epoch, total_epochs=epochs, start_time=start_time)
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
            losses = []
            for yhat, y in zip(yhats, ys):
                # Take sum of the loss of all outputs(number of outputs previous layer=inputs this layer)
                loss = sum(self.loss(yhat[o], y[o]) for o in range(self.inputs))
                losses.append(loss)
            # calculate gradients for training
            if alpha:
                gradient_vector_list = [
                    [derivative(self.loss)(yhat_i, y_i) for yhat_i, y_i in zip(yhat, y)]
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
