def train_loop(model, dataloader, criterion, optimizer, n_epochs=20, save_model=False, load_model=True):
    model.train()

    size = len(dataloader.dataset)
    best_loss = 1

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

            if batch % 100 == 0:
                test_loss = loss.item()
                current = (batch + 1) * len(X)

                print(f'Epoch {epoch}  Current {current} / {size}  Loss {test_loss}')