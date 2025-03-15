"""
Dataset handling utilities for the Fluctlight Transformer.

This module provides dataset management for byte-level transformer training:
1. Base64-encoded data loading and decoding
2. Efficient sequence collation and padding
3. Dynamic CPU worker allocation
4. Device placement optimization

The dataset is designed to work with the minimal context window of 2,
but supports arbitrary context sizes for experimentation. All sequences
are automatically padded or truncated to match the model's context window.
"""

import base64
from pathlib import Path
from typing import List, Tuple, Union, Optional, Sequence

import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader

from fluctlight.utils import decode_base64_pair

def get_num_cpu_workers(reserved_workers: int = 1) -> int:
    """
    Returns the optimal number of DataLoader workers.
    
    Calculates workers as: total_cpu_cores - reserved_workers,
    ensuring some CPU resources remain available for system processes
    while maximizing data loading efficiency.

    Args:
        reserved_workers: Number of CPU cores to reserve (default: 1)
        
    Returns:
        int: Number of usable CPU workers (minimum 1)
    """
    return max(1, multiprocessing.cpu_count() - reserved_workers)

def get_default_device() -> torch.device:
    """
    Get the optimal available device for tensor operations.
    
    Checks devices in order of computational efficiency:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        torch.device: Best available compute device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

class Base64Dataset(Dataset):
    """
    Dataset for handling Base64-encoded input-output pairs.
    
    Designed for the Fluctlight Transformer's byte-level training,
    this dataset:
    1. Loads Base64-encoded sequence pairs
    2. Supports optional training data prepending
    3. Handles device placement automatically
    4. Provides efficient sequence collation
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        prepend: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the dataset.

        Args:
            file_path: Path to data file with Base64-encoded pairs
            device: Device to place tensors on (default: auto-detect)
            prepend: Optional training data to prepend (e.g., ASCII cycles)
        
        Raises:
            FileNotFoundError: If file_path doesn't exist
            ValueError: If file contains invalid Base64 data
        """
        self.file_path = Path(file_path)
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.device = device if device is not None else get_default_device()
        
        # Prepend optional training data, such as the extended ASCII cycle 
        if prepend:
            for line in prepend:
                self.data.append(self.create_tensor_pair(*decode_base64_pair(line)))

        # Load and decode file data
        with open(self.file_path, 'r') as f:
            for line in f:
                self.data.append(self.create_tensor_pair(*decode_base64_pair(line)))
                 
    @staticmethod
    def convert_to_tensor(line: str, device: torch.device) -> torch.Tensor:
        """
        Convert a string to a tensor of byte values.
        
        Args:
            line: Input string to convert
            device: Device to place tensor on
            
        Returns:
            torch.Tensor: Long tensor of byte values
        """
        return torch.tensor([b for b in line], dtype=torch.long, device=device)
   
    def create_tensor_pair(
        self,
        input_str: str,
        target_str: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input-target tensor pair from strings.
        
        Args:
            input_str: Input sequence string
            target_str: Target sequence string
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors
        """
        return (
            Base64Dataset.convert_to_tensor(input_str, self.device),
            Base64Dataset.convert_to_tensor(target_str, self.device)
        )

    def __len__(self) -> int:
        """Return the number of sequence pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors
            
        Raises:
            IndexError: If idx is out of range
        """
        return self.data[idx]

def create_dataloader(
    dataset: Dataset,
    context_window: int,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create an optimized DataLoader for sequence training.

    The loader handles:
    1. Sequence padding to context window size
    2. Efficient batch collation
    3. Multi-worker data loading
    4. Memory pinning for GPU training

    Args:
        dataset: The dataset to load
        context_window: Size of model's context window
        batch_size: Number of sequences per batch
        shuffle: Whether to randomize sequence order
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (useful for GPU)
        persistent_workers: Keep workers alive between epochs

    Returns:
        DataLoader: Configured loader for sequence training
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_sequences(batch, context_window),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

def collate_sequences(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    context_window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate and pad sequences to uniform length.
    
    Handles sequences for the Fluctlight Transformer by:
    1. Right-aligning sequences within context window
    2. Zero-padding shorter sequences
    3. Truncating longer sequences
    
    Args:
        batch: List of (input, target) tensor pairs
        context_window: Maximum sequence length
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded input and target batches
    """
    inputs, targets = zip(*batch)
    
    # Get device from the first tensor
    device = inputs[0].device if inputs else torch.device('cpu')
    
    # Create padding tensors with the context window size
    input_padded = torch.zeros((len(inputs), context_window), dtype=torch.long, device=device)
    target_padded = torch.zeros((len(inputs), context_window), dtype=torch.long, device=device)
    
    # Fill tensors, right-aligned with zero padding
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        # Truncate or use full input based on context window
        input_seq = inp[-context_window:] if inp.size(0) > context_window else inp
        target_seq = tgt[-context_window:] if tgt.size(0) > context_window else tgt
        
        # Right-align the sequences
        input_padded[i, -input_seq.size(0):] = input_seq
        target_padded[i, -target_seq.size(0):] = target_seq
    
    return input_padded, target_padded