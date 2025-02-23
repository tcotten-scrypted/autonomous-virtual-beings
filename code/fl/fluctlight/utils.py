"""
Utility functions for the minimal Transformer implementation.

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
    decoded = base64.b64decode(encoded).decode('ascii')
    input_str, output_str = decoded.split('\t')
    return input_str, output_str

def encode_base64_pair(input_str: str, output_str: str) -> str:
    """
    Encode an input-output pair as Base64 for training data format.

    Args:
        input_str: Input string to encode
        output_str: Output string to encode

    Returns:
        Base64-encoded string with tab separator
    """
    combined = f"{input_str}\t{output_str}"
    return base64.b64encode(combined.encode('ascii')).decode('ascii')

def generate_continuation(
    model: torch.nn.Module,
    input_str: str,
    max_length: int = 100,
    temperature: float = 1.0
) -> str:
    """
    Generate text continuation using the trained Transformer model.

    The function converts input text to byte-level tokens, runs the model
    autoregessively with temperature sampling, and converts output tokens
    back to text.

    Args:
        model: Trained Transformer model
        input_str: Input string to continue from
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated continuation string
    """
    model.eval()

    # Get the model's device
    device = next(model.parameters()).device

    # Convert input string to token tensor on the correct device
    input_tokens = torch.tensor([ord(c) for c in input_str], dtype=torch.long, device=device)
    input_tokens = input_tokens.unsqueeze(0)  # Add batch dimension

    # Generate tokens autoregressively
    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(input_tokens)
            next_token_logits = logits[0, -1, :] / temperature

            # Sample from softmax distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if we generate tab (sequence delimiter)
            if next_token.item() == 9:  # ASCII tab
                break

            # Append to sequence (ensure next_token is on the correct device)
            generated.append(next_token.item())
            input_tokens = torch.cat([input_tokens, next_token.unsqueeze(0)], dim=1)

    # Convert tokens back to string
    return ''.join(chr(t) for t in generated)