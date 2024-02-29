import numpy


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
