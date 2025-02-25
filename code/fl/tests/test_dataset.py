"""Tests for dataset handling and loading."""

import torch
import pytest
from pathlib import Path
from fluctlight.dataset import (
    Base64Dataset, 
    create_dataloader,
    get_num_cpu_workers,
    collate_sequences
)

def test_base64_dataset_basic():
    """Test basic dataset functionality with a simple example."""
    # Create a temporary file with test data
    test_data = "aGVsbG8=\td29ybGQ=\n"  # "hello\tworld" in base64
    tmp_path = Path("test_data.txt")
    tmp_path.write_text(test_data)
    
    try:
        dataset = Base64Dataset(tmp_path, device=torch.device('cpu'))
        assert len(dataset) == 1
        
        # Check first item
        input_seq, target_seq = dataset[0]
        assert isinstance(input_seq, torch.Tensor)
        assert isinstance(target_seq, torch.Tensor)
        assert input_seq.device == torch.device('cpu')
        assert target_seq.device == torch.device('cpu')
        
    finally:
        tmp_path.unlink()  # Clean up

def test_base64_dataset_prepend():
    """Test dataset with prepended data."""
    test_data = "aGVsbG8=\td29ybGQ=\n"
    prepend = ["dGVzdA==\tZGF0YQ=="]  # "test\tdata" in base64
    
    tmp_path = Path("test_data.txt")
    tmp_path.write_text(test_data)
    
    try:
        dataset = Base64Dataset(
            tmp_path,
            device=torch.device('cpu'),
            prepend=prepend
        )
        assert len(dataset) == 2  # One prepended + one from file
        
    finally:
        tmp_path.unlink()

def test_collate_sequences():
    """Test sequence collation and padding."""
    # Create sequences of different lengths
    seq1 = torch.tensor([1, 2, 3])
    seq2 = torch.tensor([4, 5])
    batch = [(seq1, seq1), (seq2, seq2)]
    
    # Collate
    input_padded, target_padded = collate_sequences(batch)
    
    # Check shapes
    assert input_padded.shape == (2, 3)  # Padded to longest sequence
    assert target_padded.shape == (2, 3)
    
    # Check padding
    assert torch.all(input_padded[0] == seq1)
    assert torch.all(input_padded[1, :2] == seq2)
    assert input_padded[1, 2] == 0  # Padding

def test_dataloader_creation():
    """Test dataloader configuration."""
    test_data = "aGVsbG8=\td29ybGQ=\n"
    tmp_path = Path("test_data.txt")
    tmp_path.write_text(test_data)
    
    try:
        dataset = Base64Dataset(tmp_path, device=torch.device('cpu'))
        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            num_workers=0,  # Force single process for testing
            pin_memory=False
        )
        
        assert dataloader.batch_size == 2
        assert dataloader.num_workers == 0
        
    finally:
        tmp_path.unlink()

def test_num_cpu_workers():
    """Test CPU worker count calculation."""
    workers = get_num_cpu_workers(reserved_workers=1)
    assert workers >= 1
    assert isinstance(workers, int)

def test_invalid_base64():
    """Test handling of invalid base64 data."""
    test_data = "invalid base64!\tinvalid\n"
    tmp_path = Path("test_data.txt")
    tmp_path.write_text(test_data)
    
    try:
        with pytest.raises(Exception):  # Should raise on invalid base64
            dataset = Base64Dataset(tmp_path, device=torch.device('cpu'))
            _ = dataset[0]  # Try to access data
    finally:
        tmp_path.unlink()
