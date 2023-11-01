import argparse

import torch
import torch.nn as nn

from .models import FeedForwardNet
from .datasets import get_dataloaders

MODEL_PATH = '/outputs/ff_net.pth'

def train_loop(model, dataloader, criterion, optimizer, n_epochs=20, save_model=False, load_model=True):
    model.train()

    size = len(dataloader.dataset)
    best_loss = 1

    if load_model:
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        criterion.load_state_dict(checkpoint['loss_state_dict'])

    for epoch in range(1, n_epochs):
        for batch, (X, y) in enumerate(dataloader):
            # Feed forward - calculate prediction and loss
            pred = model(X)
            loss = criterion(pred, y)

            # Backpropogation - set weights and biases and zero gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if loss.item() < best_loss and save_model:
                best_loss = loss.item()
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss_state_dict': criterion.state_dict()
                    },
                    MODEL_PATH,
                )

            if batch % 100 == 0:
                test_loss = loss.item()
                current = (batch + 1) * len(X)

                print(f'Epoch {epoch}  Current {current} / {size}  Loss {test_loss}')


def test_loop(model, dataloader, criterion):
    model.eval()

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches

    print(f'Total correct {correct} / {dataset_size} ||  Loss {test_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python module to train network')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training loop')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('-l', type=bool, help='Load saved model - default False')
    parser.add_argument('-s', type=bool, help='Save model - default True')

    args = vars(parser.parse_args())

    n_epochs = args['epochs'] if args['epochs'] else 10
    lr = args['lr'] if args['lr'] else 0.01

    load_model = True if args['l'] else False
    save_model = True if args['s'] else False
    
    # Declare model, loss, optim
    model = FeedForwardNet()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['loss_state_dict'])

    # Get dataloaders
    training_dataloader, test_dataloader = get_dataloaders(visualize=False)

    # Train
    train_loop(model, training_dataloader, criterion, optim, n_epochs)

    # Test
    test_loop(model, test_dataloader, criterion)
    