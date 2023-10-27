import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print("Initialising network")
net = Network([784, 30, 10])

print('Starting SGD')
net.SGD(training_data, 30, 10, 0.1, test_data=test_data)

