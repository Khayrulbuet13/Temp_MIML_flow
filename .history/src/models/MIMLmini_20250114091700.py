import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18 as resnet
from typing import Tuple

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    A simple feedforward MLP for processing numeric cell properties.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        :param input_size: Dimensionality of the input features.
        :param hidden_size: Size of the hidden layer.
        :param output_size: Output dimensionality of the MLP.
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
        :param x: Tensor of shape (batch_size, input_size).
        :return: Tensor of shape (batch_size, output_size).
        """
        return self.layers(x)


class CombinedModel(nn.Module):
    """
    Combines a ResNet18 (for image features) with an MLP (for numeric/tabular features).
    """

    def __init__(self, mlp: MLP, n_classes: int, train_resnet: bool = False):
        """
        :param mlp: Instance of the MLP class to process numeric features.
        :param n_classes: Number of classes for the final classification.
        :param train_resnet: Whether to allow training of ResNet layers.
        """
        super(CombinedModel, self).__init__()
        self.resnet18 = resnet(weights='IMAGENET1K_V1')
        self.mlp = mlp

        # Freeze or unfreeze ResNet
        for param in self.resnet18.parameters():
            param.requires_grad = train_resnet

        num_features_resnet = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()

        mlp_output_size = mlp.layers[-2].out_features  # The hidden->output layer
        combined_input_size = num_features_resnet + mlp_output_size

        self.combined = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(
        self, 
        image: torch.Tensor, 
        csv_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the combined model.

        :param image: Tensor of shape (B, 3, H, W).
        :param csv_data: Tensor of shape (B, D), e.g., (B, 3) for DI, velocity, etc.
        :return: Logits of shape (B, n_classes).
        """
        x1 = self.resnet18(image)      # Image features
        x2 = self.mlp(csv_data)        # Numeric/tabular features
        x = torch.cat((x1, x2), dim=1) # Concatenate
        return self.combined(x)
