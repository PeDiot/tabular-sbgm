from typing import List, Dict
import torch.nn as nn
import torch

def nn_constructor(architecture: Dict) -> List:
    """Constructs a neural network from a dictionary of layers and their parameters."""

    layers = []

    for block in architecture:
        for layer in architecture[block]:

            layer_name = layer[0]
            if layer_name == "Linear":
                layers.append(nn.Linear(layer[1], layer[2]))
            elif layer_name == "Dropout":
                layers.append(nn.Dropout(layer[1]))
            elif layer_name == "PReLU":
                layers.append(nn.PReLU())
            elif layer_name == "ReLU":
                layers.append(nn.ReLU())
            elif layer_name == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif layer_name == "Tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"{layer_name} not written correctly or not implemented in utils.functions.nn_config_constructor")
    
    return layers

class MLPScoreNet(nn.Module):
    """A simple multi-layer perceptron (MLP) score-based network."""

    def __init__(self, architecture: Dict):
        super().__init__()

        layers = nn_constructor(architecture)
        
        self.fc = nn.Sequential(
            *layers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x