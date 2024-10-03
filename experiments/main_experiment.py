import torch
import torch.optim as optim
import torch.nn as nn
import time
from models.generalized_cnn import GeneralizedCNN
from models.mlp import MLP
from models.train import train_model
from models.evaluate import evaluate_model
from models.utils import plot_training_loss, plot_accuracy_vs_param, compute_parameters
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# 1. Compare CNN vs MLP
print("\n===== Comparing CNN and MLP Architectures =====")

mlp_model = MLP().to('cuda')
cnn_model = GeneralizedCNN(num_filters=32, kernel_size=3, use_bn=True, use_dropout=True).to('cuda')

# Optimizers
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=1e-3)
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=1e-3)

# Timing and training both models
start_time = time.time()
print("Training MLP...")
mlp_losses = train_model(mlp_model, trainloader, optimizer_mlp, loss_fn)
mlp_time = time.time() - start_time
print(f"MLP Training Time: {mlp_time:.2f} seconds")

start_time = time.time()
print("Training CNN...")
cnn_losses = train_model(cnn_model, trainloader, optimizer_cnn, loss_fn)
cnn_time = time.time() - start_time
print(f"CNN Training Time: {cnn_time:.2f} seconds")

# Evaluate both models
mlp_accuracy = evaluate_model(mlp_model, testloader)
cnn_accuracy = evaluate_model(cnn_model, testloader)

# Plot and compare loss curves
plot_training_loss(mlp_losses, title="MLP Training Loss")
plot_training_loss(cnn_losses, title="CNN Training Loss (Kernel Size = 3)")

# Display model parameter counts
mlp_params = compute_parameters(mlp_model)
cnn_params = compute_parameters(cnn_model)

print(f"MLP Test Accuracy: {mlp_accuracy:.2f}%, Parameters: {mlp_params}")
print(f"CNN Test Accuracy: {cnn_accuracy:.2f}%, Parameters: {cnn_params}")


# 2. Vary Kernel Sizes in CNN
print("\n===== Varying Kernel Sizes in CNN =====")
kernel_sizes = [2, 3, 5, 7, 9]
kernel_results = {}
kernel_train_times = []
kernel_accuracies = []

for ks in kernel_sizes:
    print(f"Testing CNN with kernel size {ks}...")
    cnn_model = GeneralizedCNN(num_filters=32, kernel_size=ks, use_bn=True, use_dropout=True).to('cuda')
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
    
    start_time = time.time()
    losses = train_model(cnn_model, trainloader, optimizer, loss_fn)
    train_time = time.time() - start_time
    accuracy = evaluate_model(cnn_model, testloader)
    
    kernel_results[ks] = accuracy
    kernel_train_times.append(train_time)
    kernel_accuracies.append(accuracy)
    
    # Plot training loss for each kernel size
    plot_training_loss(losses, title=f"Training Loss (Kernel Size: {ks})")

# Plot test accuracy vs kernel size
plot_accuracy_vs_param(kernel_sizes, kernel_accuracies, param_name="Kernel Size")


# 3. Vary Number of Filters
print("\n===== Varying Number of Filters in CNN =====")
filter_sizes = [5, 10, 15, 20, 25]
filter_results = {}
filter_accuracies = []

for filters in filter_sizes:
    print(f"Testing CNN with {filters} filters per layer...")
    cnn_model = GeneralizedCNN(num_filters=filters, kernel_size=3, use_bn=True, use_dropout=True).to('cuda')
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)
    
    losses = train_model(cnn_model, trainloader, optimizer, loss_fn)
    accuracy = evaluate_model(cnn_model, testloader)
    
    filter_results[filters] = accuracy
    filter_accuracies.append(accuracy)
    
    # Plot training loss for each filter size
    plot_training_loss(losses, title=f"Training Loss (Filters: {filters})")

# Plot test accuracy vs number of filters
plot_accuracy_vs_param(filter_sizes, filter_accuracies, param_name="Number of Filters")

# Final output for comparison in the report
print("\nFinal Results for Kernel Sizes:")
for ks, acc in kernel_results.items():
    print(f"Kernel Size: {ks}, Test Accuracy: {acc:.2f}%")

print("\nFinal Results for Filter Sizes:")
for filters, acc in filter_results.items():
    print(f"Filters: {filters}, Test Accuracy: {acc:.2f}%")
