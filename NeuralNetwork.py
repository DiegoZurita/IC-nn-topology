import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

class NeuralNetwork():
    _biases = []
    _weights = []
    _n_layers = 0

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def add_layer(self, neurons):
        self._biases.append( np.random.randn(neurons) )

        nrows = self.input_size
        if self._n_layers > 0:
            # Neurons in the last hidden layer appeeden
            nrows = self._weights[-1].shape[1]

        self._weights.append( np.random.randn(nrows, neurons) )
        self._n_layers += 1

    def compile(self):
        # Add the weight matrix of output layer
        self.add_layer(self.output_size)
        self._n_layers += 1

    def predict(self, x_vector):
        predictions = []
        
        for x in x_vector:
            _, activations = self.feedfoward(x)
            predictions.append(activations[-1])

        return predictions

    def feedfoward(self, x):
        assert self.input_size == len(x)
        a = x
        zs = []
        activations = [a]

        for weight, bias in zip(self._weights, self._biases):
            z = np.dot(a, weight) + bias
            zs.append(z)
            a = self.activation(z)
            activations.append(a)

        return zs, activations

    def backpropagate(self, x, y):
        news_b = [ np.zeros(b.shape) for b in self._biases ]
        news_w = [ np.zeros(w.shape) for w in self._weights ]

        zs, activations = self.feedfoward(x)
        delta = self.cost_of_derivative(activations[-1], y)

        news_b[-1] = delta
        news_w[-1] = self.cost_matrix_over_weitght(activations[-2], delta)

        for l in range(2, self._n_layers):
            z = zs[-l]
            sp = self.activation_prime(z)
            # backpropagate the error
            delta = np.dot(self._weights[-l+1], delta) * sp
            news_b[-l] = delta
            news_w[-l] = self.cost_matrix_over_weitght(activations[-l-1], delta)

        return news_w, news_b

    def update_mini_batch(self, mini_batch, eta):
        news_b = [np.zeros(b.shape) for b in self._biases]
        news_w = [np.zeros(w.shape) for w in self._weights]
        
        for x, y in mini_batch:
            delta_w, delta_b = self.backpropagate(x, y)

            news_b = [ nb+db for nb, db in zip(news_b, delta_b)]
            news_w = [ nw+dw for nw, dw in zip(news_w, delta_w)]

        m = len(mini_batch)

        self._weights = [ w - (eta/m)*nw for w, nw in zip(self._weights, news_w)]
        self._biases = [ b - (eta/m)*nb for b, nb in zip(self._biases, news_b)]

    def train(self, X, y, epochs=100, eta=0.01, batch_size=30):
        print("Start training..")
        r = np.arange(len(y))
        
        costs = []

        for _ in range(epochs):
            costs.append(self.cost(X, y))
            indexes = np.random.choice(r, batch_size)
            mini_batch = [ (X[i], y[i]) for i in indexes ]
            self.update_mini_batch(mini_batch, eta)

        print("End training")
        return costs

    def cost_matrix_over_weitght(self, activation, delta):
        columns = []
        for delta_i in delta:
            columns.append(activation*delta_i)

        return np.column_stack(columns)
        
    def cost(self, X, y):
        c_cum = 0
        m = len(y)
        y_hat = self.predict(X)

        for i in range(m):
            c_cum = np.linalg.norm(y_hat[i] - y[i]) ** 2

        return c_cum/(2*m)

    def cost_of_derivative(self, output, y):
        return output - y 

    # Sigmoid
    def activation(self, z):
        return 1/(1 + np.exp(-z))

    # Sigmoid prime
    def activation_prime(self, z):
        return self.activation(z)*(1-self.activation(z))


if __name__ == "__main__":
    n = NeuralNetwork(4, 1)
    n.add_layer(2)
    n.compile()

    X, y = make_classification(n_features=4, n_samples=1000)

    _ = n.train(X, y, epochs=100, eta=2)

    y_predic = n.predict(X)
    y_predic = np.array([ round(y_hat[0]) for y_hat in y_predic])
    print("Acerto de {}".format(np.sum(y == y_predic)/len(y)))