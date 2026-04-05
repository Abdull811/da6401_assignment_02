"""Reusable custom layers 
"""

import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """
    Custom Dropout layer.
    """

    def __init__(self, p: float=0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        
        super().__init__()

        # Validate probability
        assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        
        # If in evaluation mode → no dropout
        if not self.training or self.p == 0:
            return x

        # Create Bernoulli mask (same shape as input)
        mask = (torch.rand_like(x) > self.p).float()

        # Apply inverted dropout scaling
        out = x * mask / (1 - self.p)

        return out
