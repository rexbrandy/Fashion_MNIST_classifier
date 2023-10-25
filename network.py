import numpy as np
import random

class Network():
    '''
    This class is a basic neural network. It contains functions for Sigmoid activation, a feed forward loop and 
    stocastic gradient descent.
    The biases and weights for each neuron are randomly generated Gaussian distrubtions.
    This will be the statrting point for each neuron and we hope to improve this with training.

    Parameters:
    size: array - each element is a layer, the value is the number of neuron
    '''
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        print(self.biases)
        print(self.weights)

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, batch_size, epochs, lr, test_data=None):
        n_samples = len(training_data)

        for epoch in epochs:
            random.shuffle(training_data)

            for i in range(n_samples):
                mini_batches = [training_data[k:k+batch_size] for k in range(0, n_samples, batch_size)]

            for batch in mini_batches:
                self.update_mini_batch(batch, lr)

    def update_mini_batch(self, batch, lr):
        nabla_b = [np.zeroes(b.shape) for b in self.biases]
        nabla_w = [np.zeroes(w.shape) for w in self.weights]

        for X, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(X, y)

            nabla_b = [(nb+dnb) for nb, dnb, in zip(nabla_b, delta_nabla_b)]
            nabla_w = [(nw+dnw) for nw, dnw, in zip(nabla_w, delta_nabla_w)]

    def backprop(self, X, y):
        nabla_b = [np.zeroes(b.shape) for b in self.biases]
        nabla_w = [np.zeroes(w.shape) for w in self.weights]

        activation = X
        activations = [X]

        z_vectors = []

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            z_vectors.apped(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_vectors[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = z_vectors[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(X)), y) for (X, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))