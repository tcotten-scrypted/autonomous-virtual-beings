"""
Test device handling and tensor placement in the model and dataset.
"""

import torch
from fluctlight.model import FluctlightTransformer, get_default_device
from fluctlight.dataset import Base64Dataset

def test_device_detection():
    """Test automatic device detection and tensor placement."""
    # Get default device
    device = get_default_device()
    
    # Initialize model
    model = FluctlightTransformer(device=device)
    assert model.device == device, "Model not on correct device"
    
    # Create a sample input
    x = torch.randint(0, 256, (1, 10), device=device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Verify all tensors are on same device
    assert x.device == model.device, "Input tensor not on model device"
    assert output.device == model.device, "Output tensor not on model device"

def test_explicit_device_placement():
    """Test explicit device placement works."""
    cpu_device = torch.device('cpu')
    model = FluctlightTransformer(device=cpu_device)
    
    # Verify model is on CPU
    assert model.device == cpu_device
    assert next(model.parameters()).device == cpu_device
    
    # Test forward pass maintains device
    x = torch.randint(0, 256, (1, 10), device=cpu_device)
    with torch.no_grad():
        output = model(x)
    assert output.device == cpu_device

if __name__ == "__main__":
    test_device_detection()
    test_explicit_device_placement()
