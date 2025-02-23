"""
Test device handling and tensor placement in the model and dataset.
"""

import torch
from fluctlight.model import MinimalTransformer, get_default_device
from fluctlight.dataset import Base64Dataset

def test_device_detection():
    """Test automatic device detection and tensor placement."""
    # Get default device
    device = get_default_device()
    print(f"\nDetected device: {device}")
    
    # Initialize model
    model = MinimalTransformer(device=device)
    print(f"Model device: {model.device}")
    
    # Create a sample input
    x = torch.randint(0, 256, (1, 10), device=device)
    print(f"Input tensor device: {x.device}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    print(f"Output tensor device: {output.device}")
    
    # Verify all tensors are on same device
    assert x.device == model.device, "Input tensor not on model device"
    assert output.device == model.device, "Output tensor not on model device"
    
    # Test dataset device handling
    dataset = Base64Dataset("data/train.txt", device=device)
    sample = dataset[0]
    print(f"Dataset sample device: {sample.device}")
    assert sample.device == device, "Dataset tensor not on correct device"
    
    print("\nAll device placement tests passed!")

if __name__ == "__main__":
    test_device_detection()
