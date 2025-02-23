"""
Tests for the dataset handling utilities.
"""

import base64
import pytest
import torch
from pathlib import Path

from fluctlight.dataset import Base64Dataset, create_dataloader, collate_sequences

@pytest.fixture
def sample_data_file(tmp_path):
    """Create a temporary file with sample data."""
    data = [
        "SGVsbG8JV29ybGQ=",  # Hello\tWorld
        "VGVzdAlEYXRh"       # Test\tData
    ]
    
    file_path = tmp_path / "test_data.txt"
    with open(file_path, 'w') as f:
        f.write('\n'.join(data))
    
    return file_path

def test_dataset_loading(sample_data_file):
    """Test dataset loading and decoding."""
    dataset = Base64Dataset(sample_data_file)
    
    assert len(dataset) == 2
    
    # Check first example
    first_item = dataset[0]
    assert isinstance(first_item, torch.Tensor)
    assert first_item.dtype == torch.long
    
    # Decode and verify content
    decoded = ''.join(chr(x) for x in first_item.tolist())
    assert decoded == "Hello\tWorld"

def test_dataloader_creation(sample_data_file):
    """Test DataLoader creation and batch loading."""
    dataset = Base64Dataset(sample_data_file)
    dataloader = create_dataloader(dataset, batch_size=2)
    
    # Get first batch
    batch = next(iter(dataloader))
    
    assert isinstance(batch, torch.Tensor)
    assert batch.dim() == 2
    assert batch.size(0) == 2  # batch size

def test_sequence_collation():
    """Test sequence collation with padding."""
    # Create sequences of different lengths
    sequences = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([8, 9])
    ]
    
    # Collate
    batch = collate_sequences(sequences)
    
    assert batch.shape == (3, 4)  # 3 sequences, max length 4
    assert torch.all(batch[0, :3] == torch.tensor([1, 2, 3]))
    assert torch.all(batch[0, 3:] == 0)  # padding
    assert torch.all(batch[1] == torch.tensor([4, 5, 6, 7]))
    assert torch.all(batch[2, :2] == torch.tensor([8, 9]))
    assert torch.all(batch[2, 2:] == 0)  # padding
