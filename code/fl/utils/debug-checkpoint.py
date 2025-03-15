"""
Simple script to inspect a Fluctlight checkpoint's configuration.
Usage: python test_checkpoint.py path/to/checkpoint.ckpt
"""

import sys
import torch
from pprint import pprint

def inspect_checkpoint(checkpoint_path: str) -> None:
    """
    Load and inspect a checkpoint file's configuration.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n=== Checkpoint Structure ===")
        print("Top-level keys:", list(checkpoint.keys()))
        
        print("\n=== State Dict Keys ===")
        if "state_dict" in checkpoint:
            print("State dict keys:", list(checkpoint["state_dict"].keys()))
        
        print("\n=== Configuration ===")
        if "config" in checkpoint["state_dict"]:
            print("From state_dict['config']:")
            pprint(checkpoint["state_dict"]["config"])
        
        print("\nHyperparameters:")
        if "hyper_parameters" in checkpoint:
            pprint(checkpoint["hyper_parameters"])
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_checkpoint.py path/to/checkpoint.ckpt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path)
