"""
Dataset handling utilities for the Fluctlight Transformer.
"""

import base64
from pathlib import Path
from typing import List, Tuple, Union, Optional

import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader

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
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

class Base64Dataset(Dataset):
    """Dataset for handling Base64-encoded input-output pairs."""

    def __init__(
        self,
        file_path: Union[str, Path],
        device: Optional[torch.device] = None,
        prepend: Optional[List[str]] = None
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
        return torch.tensor([b for b in line], dtype=torch.long, device=device)
   
    def create_tensor_pair(self, input, target):
        return (
            Base64Dataset.convert_to_tensor(input, self.device),
            Base64Dataset.convert_to_tensor(target, self.device)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        """
        Get a single example from the dataset.

        Returns:
            Tensor of byte values representing the sequence
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
        collate_fn=lambda batch: collate_sequences(batch, context_window),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

def collate_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor]], context_window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for padding sequences to the same length, respecting max context window.
    Total sequence length (input + target shift) should not exceed MAX_CONTEXT.
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