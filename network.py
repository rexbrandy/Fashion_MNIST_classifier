import numpy as np

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

    def sigmoid(z):
        '''
        This is a sigmoid activation function.

        Parameters:
        z: TODO - weighted sum of paramters
        '''
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a


if __name__ == '__main__':
    net = Network([2, 4, 1])

    print('Biases')
    print(net.biases)
    print()
    print('Weights')
    print(net.weights)