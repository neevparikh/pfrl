"""Initializes the weights and biases of a layer to chainer default.
"""
import torch
import torch.nn as nn

@torch.no_grad()
def init_xavier_uniform(layer, nonlinearity='relu'):
    """Initializes the layer with xavier_uniform 
    """
    assert isinstance(layer, nn.Module)

    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain(nonlinearity))
        if layer.bias is not None:
            # layer may be initialized with bias=False
            nn.init.zeros_(layer.bias)
    return layer
