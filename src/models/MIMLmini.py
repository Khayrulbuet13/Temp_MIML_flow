import logging
import torch
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """
    A simple feedforward MLP for processing numeric cell properties.
    Given an input of shape (batch_size, input_size), this MLP
    returns features of shape (batch_size, output_size).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        :param input_size: Dimensionality of the input numeric features.
        :param hidden_size: Size of the hidden layer.
        :param output_size: Final output dimensionality of the MLP.
        """
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        :param x: Tensor of shape (batch_size, input_size).
        :return: Tensor of shape (batch_size, output_size).
        """
        return self.layers(x)


class TinyBackbone(nn.Module):
    """
    A small CNN backbone that processes single-channel images of size (1, 48, 48).
    It outputs a 1D feature vector per image (flattened).
    """
    def __init__(self, input_size=(1, 48, 48)):
        """
        :param input_size: (channels, height, width) of the input image.
        """
        super(TinyBackbone, self).__init__()
        # Define a small CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dynamically compute the number of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)  # shape (1, 1, 48, 48)
            dummy_output = self.features(dummy_input)
            self.num_features = dummy_output.view(-1).size(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the tiny CNN.

        :param x: Image tensor of shape (batch_size, 1, 48, 48).
        :return: Flattened feature tensor of shape (batch_size, self.num_features).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the spatial dimensions
        return x


class CombinedModel(nn.Module):
    """
    Combines a tiny CNN (for single-channel image features)
    with an MLP (for numeric features) and outputs class logits.
    """
    def __init__(self, mlp: MLP, n_classes: int, train_cnn: bool = True):
        """
        :param mlp: Instance of the MLP class to process numeric features.
        :param n_classes: Number of classes for the final classification.
        :param train_cnn: Whether to allow training of the CNN backbone.
        """
        super(CombinedModel, self).__init__()

        # Tiny CNN Backbone
        self.tiny_cnn = TinyBackbone(input_size=(1, 48, 48))

        # Freeze or unfreeze the tiny CNN (if desired)
        for param in self.tiny_cnn.parameters():
            param.requires_grad = train_cnn

        # MLP for numeric/tabular features
        self.mlp = mlp

        # Determine feature sizes
        cnn_output_size = self.tiny_cnn.num_features
        mlp_output_size = mlp.layers[-2].out_features  # The hidden->output layer of the MLP

        # Linear layers to combine CNN + MLP features
        combined_input_size = cnn_output_size + mlp_output_size
        self.combined = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, image: torch.Tensor, csv_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the combined model.

        :param image: Image tensor of shape (batch_size, 1, 48, 48).
        :param csv_data: Numeric tensor of shape (batch_size, D).
        :return: Logits of shape (batch_size, n_classes).
        """
        # Extract image features from tiny CNN backbone
        x1 = self.tiny_cnn(image)

        # Extract numeric features from MLP
        x2 = self.mlp(csv_data)

        # Concatenate image features and numeric features
        x = torch.cat((x1, x2), dim=1)

        # Pass the concatenated features through the final classifier
        return self.combined(x)
