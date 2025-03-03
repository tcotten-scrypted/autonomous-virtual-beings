"""
Utility functions for the Fluctlight Transformer implementation.

This module provides utilities for:
1. Base64 encoding/decoding of training data
2. Text generation with temperature sampling
3. Data processing helpers
"""

import base64
from typing import Tuple, List, Optional

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


def generate_continuation(
    model: torch.nn.Module,
    input_str: str,
    max_length: Optional[int] = None,
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
    
    max_length = max_length or max(1, model.context_window - len(input_str))

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
            
            # Normalize logits before softmax to prevent extreme probability imbalances
            next_token_logits = logits[0, -1, :]
            next_token_logits = next_token_logits - next_token_logits.max()  # Subtract max value for numerical stability

            # Apply temperature scaling
            probs = torch.softmax(next_token_logits / temperature, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence (ensure next_token is on the correct device)
            generated.append(next_token.item())
            
            # Ensure input_tokens does not exceed CONTEXT_WINDOW
            next_token_reshaped = next_token.view(1, 1)  # Reshape to [1, 1] for batch_size and seq_len
            input_tokens = torch.cat([input_tokens, next_token_reshaped], dim=1)
            
            # Truncate oldest tokens, preserving the last `context_window` tokens
            if input_tokens.shape[1] > model.context_window:
                input_tokens = input_tokens[:, -model.context_window:]

    # Convert tokens back to string
    return ''.join(chr(t) for t in generated)