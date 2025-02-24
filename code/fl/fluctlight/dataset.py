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

from fluctlight.utils import decode_base64_pair

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
                a, b = decode_base64_pair(line)
                self.data.append(self.create_tensor_pair(a, b))

        # Load and decode data
        with open(self.file_path, 'r') as f:
            for line in f:
                a, b = decode_base64_pair(line)
                self.data.append(self.create_tensor_pair(a, b))
                    
    def convert_to_tensor(self, line):
        return torch.tensor([b for b in line], dtype=torch.long, device=self.device)
    
    def create_tensor_pair(self, a, b):
        return (
            self.convert_to_tensor(a),
            self.convert_to_tensor(b)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example from the dataset.

        Returns:
            Tensor of byte values representing the sequence
        """

        return self.data[idx]

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

def collate_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for padding input and target sequences to the same length.

    Args:
        batch: List of (input_tensor, target_tensor) tuples.

    Returns:
        Tuple of padded tensors (inputs, targets) with shape [batch_size, max_seq_len].
    """
    inputs, targets = zip(*batch)  # Separate input and target sequences

    # Find max length in batch
    max_len = max(seq.size(0) for seq in inputs + targets)  # Consider both inputs and targets

    # Get device from the first tensor
    device = inputs[0].device if inputs else torch.device('cpu')

    # Create padding tensors
    input_padded = torch.full((len(inputs), max_len), 0, dtype=torch.long, device=device)
    target_padded = torch.full((len(targets), max_len), 0, dtype=torch.long, device=device)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        input_padded[i, :inp.size(0)] = inp
        target_padded[i, :tgt.size(0)] = tgt

    return input_padded, target_padded
