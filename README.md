# Fashion MNIST Classifier
Theis module provides a simple example of building and training a neural network for image classification using the Fashion MNIST dataset. The trained model can classify images of clothing into one of ten categories, including items like T-shirts, trousers, dresses, sneakers, and more.

# Requirements
Before using the script, ensure you have the following dependencies installed:

Python 3.9
PyTorch (usually installed with pip install torch)
torchvision
NumPy
Matplotlib (for visualization, optional)

# Usage
To train model
```
python train.py
```

To make predictions
```
python main.py
```

# Model Architecture
The neural network used for this classification task consists of several fully connected (linear) layers with ReLU activation functions. The input layer has the same number of neurons as the number of pixels in the Fashion MNIST images (28x28), and the output layer has 10 neurons, each representing a clothing category.

# Training
The `train.py` script will train the model on the Fashion MNIST dataset. The dataset is automatically downloaded and split into training and testing sets. Training parameters such as batch size, learning rate, and number of epochs can be adjusted in the script.
```
  --epochs EPOCHS  Number of epochs for training loop
  --lr LR          Learning rate
  -l L             Load saved model
  -s S             Save model
```