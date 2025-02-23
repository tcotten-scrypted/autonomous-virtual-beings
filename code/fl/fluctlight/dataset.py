"""
Dataset handling utilities for the minimal Transformer.
"""

import base64
from pathlib import Path
from typing import List, Tuple, Union, Optional

import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader

import multiprocessing

def get_num_cpu_workers(reserved_workers=1):
    """
    Returns the number of recommended DataLoader workers based on:
    `total_cpu_cores - reserved_workers`, ensuring at least one worker.
    
    The `reserved_workers` parameter ensures some CPU resources are left
    available for system processes.

    Returns:
        int: Number of usable CPU workers.
    """
    return max(1, multiprocessing.cpu_count() - reserved_workers)

def get_default_device() -> torch.device:
    """
    Get the optimal available device (Metal, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Base64Dataset(Dataset):
    """Dataset for handling Base64-encoded input-output pairs."""

    def __init__(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        prepend: Optional[List[str]] = []
    ):
        """
        Initialize the dataset.

        Args:
            file_path: Path to the data file containing Base64-encoded pairs
            device: Device to place tensors on (default: None, auto-detect)
        """
        self.file_path = Path(file_path)
        self.data: List[str] = []
        self.device = device if device is not None else get_default_device()
        
        # Prepend optional training data, such as the ASCII cycle 
        if prepend:
            for line in prepend:
                line = line.strip()
                if line:
                    self.data.append(line)

        # Load and decode data
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.data.append(line)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single example from the dataset.

        Returns:
            Tensor of byte values representing the sequence
        """
        # Decode Base64 string
        encoded = self.data[idx]
        decoded = base64.b64decode(encoded)

        # Convert bytes to tensor on correct device
        return torch.tensor([b for b in decoded], dtype=torch.long, device=self.device)

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: The dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes,
        pin_memory: Beneficial for GPUs
        persistent_workers: Performance improvement in PyTorch 1.7+

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sequences,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

def collate_sequences(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Collate function for padding sequences to same length.

    Args:
        batch: List of sequence tensors

    Returns:
        Padded tensor of shape [batch_size, max_seq_len]
    """
    # Find max length in batch
    max_len = max(seq.size(0) for seq in batch)

    # Get device from first tensor in batch
    device = batch[0].device if batch else torch.device('cpu')

    # Pad sequences on the correct device
    padded = torch.full((len(batch), max_len), 0, dtype=torch.long, device=device)
    for i, seq in enumerate(batch):
        padded[i, :seq.size(0)] = seq

    return padded