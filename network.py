import numpy as np

class Network():
    '''
    This class will take a list as an argument.
    Each element in the list will represent a layer in the network and the value of the element will
    represent the number of neurons in each layer.

    net = Network([2, 4, 1])

    This input would represent 3 layers.
    Layer 1 would contain 2 neurons
    Layer 2 contains 4 neurons.
    Layer 3 contains 1 final neuron.

    The biases and weights for each neuron are randomly generated Gaussian distrubtions.
    This will be the statrting point for each neuron and we hope to improve this with training.
    '''
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        print(self.biases)
        print(self.weights)

    def sigmoid(z):
        return 1 / 1 + np.exp(-z)

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