import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        # conv1 -> relu -> batchnorm -> conv2 -> relu -> batchnorm -> flatten -> dropout -> fc1 -> relu -> dropout -> fc2
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for conv1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization for conv2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Add batch normalization after conv1
        x = self.relu(self.bn2(self.conv2(x)))  # Add batch normalization after conv2
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))  # Add dropout after fc1
        x = self.fc2(x)  # Output raw logits
        return x