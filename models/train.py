def train_model(model, train_loader, optimizer, loss_fn, n_epochs=5):
    """
    Train the given model using the specified DataLoader, optimizer, and loss function.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        loss_fn (torch.nn.Module): Loss function to compute training loss.
        n_epochs (int): Number of training epochs.
    """
    model.train()
    train_loss = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        train_loss.append(avg_loss)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    return train_loss