"""
Utility functions for the Fluctlight Transformer implementation.

This module provides utilities for:
1. Base64 encoding/decoding of training data
2. Text generation with temperature sampling
3. Data processing helpers
"""

import base64
from typing import Tuple, List

import torch

def decode_base64_pair(encoded: str) -> Tuple[str, str]:
    """
    Decode a Base64-encoded input-output pair from training data.

    The training data consists of tab-separated input-output pairs
    encoded in Base64 format. This function decodes a single pair.

    Args:
        encoded: Base64-encoded string containing tab-separated pair

    Returns:
        Tuple of (input_string, output_string)
    """
    input_str, output_str = encoded.split('\t')
    input_str = base64.b64decode(input_str)
    output_str = base64.b64decode(output_str)
    
    return input_str, output_str