import numpy as np # Not used directly, but often present with torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # Requires einops to be installed
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x (according to original comment)
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    # Ensure 3D for snake op if it's not already
    is_1d_input = False
    if x.ndim == 2: # (B, L) or (C, L) - assuming (B,L) for typical nn processing
        x = x.unsqueeze(1) # Treat as (B, 1, L)
        is_1d_input = True
    elif x.ndim != 3: # Not (B, C, L)
        # This case should ideally not happen if inputs are shaped correctly
        # For safety, could raise error or try to reshape if logic is clear
        pass

    # Original reshape logic assumes x is at least 2D (B, C, ...)
    # x_reshaped = x.reshape(shape[0], shape[1], -1) # This line was problematic if shape had <2 dims
    # For (B, C, L), shape[0]=B, shape[1]=C, -1 covers L

    # Simpler approach for (B, C, L) which snake expects for alpha application
    # No need to reshape if x is already (B,C,L) and alpha is (1,C,1) or (B,C,1)

    # The original code's reshape:
    # shape = x.shape
    # x = x.reshape(shape[0], shape[1], -1)
    # This is only safe if x.ndim >= 2. If x.ndim == 3 (B,C,L), shape[0]=B, shape[1]=C, then -1 makes the last dim L.
    # If x.ndim == 2 (B,L) and alpha is (1,1,1) after unsqueeze, this would be shape[0]=B, shape[1]=1, then -1 makes last dim L.

    # Let's assume x is [B, C, T] and alpha is broadcastable (e.g. [1, C, 1])
    # The original snake function's reshape was mainly to flatten spatial/temporal dims if they existed beyond T.
    # For 1D Conv a la DAC, T is the only spatial/temporal dim.

    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)

    if is_1d_input and x.ndim == 3 and x.shape[1] == 1: # If we added a channel dim
        x = x.squeeze(1) # Remove it back to (B,L)
    # x = x.reshape(shape) # Restore original shape if it was more complex
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Initialize alpha as a learnable parameter, matching input channels
        # Shape [1, C, 1] for broadcasting over [B, C, T]
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        # x is expected to be [B, C, T]
        return snake(x, self.alpha)
