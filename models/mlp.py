import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 1000)  # Input layer
        self.fc2 = nn.Linear(1000, 10)  # Output layer (10 classes)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.relu(self.fc1(x))  # Hidden layer
        x = self.fc2(x)  # Output layer
        return x