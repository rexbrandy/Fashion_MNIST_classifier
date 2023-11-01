import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .models import FeedForwardNet
from .datasets import get_dataloaders
from .train import train_loop, test_loop
from .utils import labels_map

MODEL_PATH = '/outputs/ff_net.pth'

# Declare model, loss, optim
model = FeedForwardNet()

# Load checkpoint
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

# Get dataloaders
training_dataloader, test_dataloader = get_dataloaders(visualize=False)

X, y = next(iter(training_dataloader))

image, correct_label = X[0], y[0]

guess = labels_map[int(model(image).argmax(1)[0])]
answer = labels_map[int(correct_label)]

print(f'Prediction: {guess}')
print(f'Answer: {answer}')
plt.imshow(image.squeeze(), cmap='gray')