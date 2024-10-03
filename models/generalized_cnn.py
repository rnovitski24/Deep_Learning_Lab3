import torch
import torch.nn as nn

class GeneralizedCNN(nn.Module):
    """
    Generalized CNN model where parameters such as number of filters, kernel size, Batch Normalization,
    and Dropout can be adjusted for flexible experimentation.  This allows for one model to be used for all
    desired experiments to reduce any redundancies.

    Args:
        num_filters (int): Number of filters for the first convolution layer
        kernel_size (int): Size of the convolution kernel.
        use_bn (bool): Whether to use Batch Normalization after convolution layers.
        use_dropout (bool): Whether to apply Dropout after certain layers.
        dropout_prob (float): Dropout probability to use if Dropout is applied.
    """
    def __init__(self, num_filters=32, kernel_size=3, use_bn=False, use_dropout=False, dropout_prob=0.5):
        super(GeneralizedCNN, self).__init__()
        
        # Padding to preserve spatial dimensions
        padding = (kernel_size - 1) // 2

        # First Convolutional Block: Conv -> (Optional BN) -> (Optional Dropout)
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_filters) if use_bn else None
        self.dropout1 = nn.Dropout(dropout_prob) if use_dropout else None

        # Second Convolutional Block: Conv -> (Optional BN) -> Pooling
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_filters * 2) if use_bn else None
        self.pool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by half

        # Third Convolutional Block: Conv -> (Optional BN) -> Pooling
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(num_filters * 4) if use_bn else None
        self.pool2 = nn.MaxPool2d(2, 2)  # Further reduces spatial dimensions

        # Dummy input to calculate the flattened feature size
        self.example_input = torch.randn(1, 1, 28, 28)
        self.flatten_size = self._get_flatten_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 200)  # First fully connected layer
        self.fc2 = nn.Linear(200, 10)  # Output layer for 10 classes (FashionMNIST)

    def _get_flatten_size(self):
        """
        Helper function to calculate the size of the flattened feature map after convolution and pooling.
        This is needed to set up the input size for the first fully connected layer.
        
        Returns:
            int: The total number of elements in the flattened feature map.
        """
        x = torch.relu(self.conv1(self.example_input))
        if self.bn1: x = self.bn1(x)
        if self.dropout1: x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        if self.bn2: x = self.bn2(x)
        x = torch.relu(self.conv3(x))
        if self.bn3: x = self.bn3(x)
        x = self.pool2(x)
        return x.numel()  # Return the number of elements in the flattened tensor

    def forward(self, x):
        """
        Forward pass for the CNN model.

        Args:
            x (torch.Tensor): Input tensor (image data).

        Returns:
            torch.Tensor: The output logits for classification.
        """
        x = torch.relu(self.conv1(x))
        if self.bn1: x = self.bn1(x)
        if self.dropout1: x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        if self.bn2: x = self.bn2(x)
        x = torch.relu(self.conv3(x))
        if self.bn3: x = self.bn3(x)
        x = self.pool2(x)
        x = x.view(-1, self.flatten_size)  # Flatten before fully connected layers
        x = torch.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x
