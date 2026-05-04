import torch
from torch.optim import Adam

def get_optimizer(model, lr=0.001):
    """
    Initialize the Adam optimizer for the given model.
    
    Args:
    - model (nn.Module): The neural network model.
    - lr (float): Learning rate for the optimizer.

    Returns:
    - optimizer: The Adam optimizer.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    return optimizer