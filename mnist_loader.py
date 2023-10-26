import pickle
import gzip

import numpy as np

def load_file(path):
    '''
    Open MNIST data file and return training, validation and test data

    Outputs:
    `training_data` Tuple with 2 elements.

        0th element numpy.nparray with shape = (50000, 784).
        This is the training inputs where each input is an ndarray 
        representing the 28 * 28 = 784 pixels in a single MNIST image 

        1st elemnent numpy.nparray with shape = (50000,).
        This is labels for the training inputs. each label is a single int 
        from 0-9 this is the corresponding number for each input.

    `test_data` and `validation_data` have the same shape but with only 10000
    images, to be used for evaluation
    '''
    with gzip.open(path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    return (training_data, validation_data, test_data)

def vectorize_results(j):
    results = np.zeroes((10, 1))
    results[j] = 1.0
    return results

def load_data_wrapper():
    training, validation, test = load_file('data/mnist.pkl.gz')    

    training_inputs = [np.reshape(X, (784, 1)) for X in training[0]]
    training_results = [vectorize_results(y) for y in training[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(X, (784, 1)) for X in validation[0]]
    validation_data = zip(validation_inputs, validation[1])

    test_inputs = [np.reshape(X, (784, 1)) for X in test[0]]
    test_data = zip(test_inputs, test[1])

    return (training_data, validation_data, test_data)

if __name__ == '__main__':
    (training_data, validation_data, test_data) = load_file('data/mnist.pkl.gz')
    print('training:\n', training_data[0].shape)
    print('validation:\n', validation_data[0].shape)
    print('test:\n', test_data[0].shape)
    print()
    print('training:\n', training_data[1].shape)
    print('validation:\n', validation_data[1].shape)
    print('test:\n', test_data[1].shape)
    