import matplotlib.pyplot as plt

def plot_training_loss(losses, title="Training Loss"):
    """Plot training loss over epochs."""
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_vs_param(param_values, accuracies, param_name='Kernel Size'):
    """Plot accuracy vs a given parameter."""
    plt.figure(figsize=(8, 6))
    plt.plot(param_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title(f'Test Accuracy vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.show()

def compute_parameters(model):
    """Calculate the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
