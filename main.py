import torch
import torch.nn as nn

from .models import FeedForwardNet
from.datasets import get_dataloaders
from .train import train_loop
from .tests import test_loop

MODEL_PATH = '/outputs/ff_net.pth'

n_epochs = 50
lr = 0.01

# Declare model, loss, optim
model = FeedForwardNet()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
criterion.load_state_dict(checkpoint['loss_state_dict'])

# Get dataloaders
training_dataloader, test_dataloader = get_dataloaders(visualize=True)

# Train
train_loop(model, training_dataloader, criterion, optim, n_epochs)

# Test
test_loop(model, test_dataloader, criterion)

# Save
total_epochs = n_epochs + checkpoint['epoch']

torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'loss_state_dict': criterion.state_dict()
    },
    MODEL_PATH,
)