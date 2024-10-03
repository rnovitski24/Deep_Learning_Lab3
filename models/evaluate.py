import torch

@torch.no_grad()
def evaluate_model(model, test_loader):
    """
    Evaluate the given model on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
    """
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get predicted labels
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # Count correct predictions
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy