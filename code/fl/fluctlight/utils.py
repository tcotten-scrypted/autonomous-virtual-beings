"""
Utility functions for the Fluctlight Transformer implementation.

This module provides core utilities for:
1. Base64 encoding/decoding of training data
2. Text generation with temperature sampling
3. Data processing helpers
4. Model loading and testing utilities

The utilities are optimized for the minimal context window of 2 tokens,
but support arbitrary context sizes. Temperature sampling is particularly
important for controlling pattern generation in the minimal context case.
"""

import base64
import math
import csv
from typing import Tuple, List, Optional, Union, cast

import torch
import torch.nn as nn

def decode_base64_pair(encoded: str) -> Tuple[bytes, bytes]:
    """
    Decode a Base64-encoded input-output pair from training data.

    The training data consists of tab-separated input-output pairs
    encoded in Base64 format. This function decodes a single pair.

    Args:
        encoded: Base64-encoded string containing tab-separated pair

    Returns:
        Tuple[bytes, bytes]: Decoded (input_bytes, output_bytes) pair
        
    Raises:
        ValueError: If input lacks tab separator
        binascii.Error: If Base64 decoding fails
    """
    try:
        input_str, output_str = encoded.strip().split('\t')
        input_bytes = base64.b64decode(input_str)
        output_bytes = base64.b64decode(output_str)
        return input_bytes, output_bytes
    except ValueError as e:
        raise ValueError("Invalid format: input must be tab-separated") from e
    except base64.binascii.Error as e:
        raise base64.binascii.Error("Invalid Base64 encoding") from e


def generate_continuation(
    model: nn.Module,
    input_str: str,
    max_length: Optional[int] = None,
    temperature: float = 1.0,
    verbose: bool = False
) -> str:
    """
    Generate text continuation with temperature-controlled sampling.
    
    This function is optimized for the minimal context window (2 tokens),
    where temperature control is crucial for pattern stability. Lower
    temperatures (0.1-0.3) tend to produce more stable patterns, while
    higher temperatures introduce more variation.

    Args:
        model: The Fluctlight model to use for generation
        input_str: Seed text for generation
        max_length: Maximum length to generate (default: context_window)
        temperature: Sampling temperature (default: 1.0)
            - 0.1-0.3: Stable patterns, good for testing
            - 0.4-0.7: Balanced variation
            - 0.8-1.0: Maximum variation
        verbose: Whether to print debug information

    Returns:
        str: Generated text continuation
        
    Raises:
        ValueError: If temperature <= 0
        AttributeError: If model lacks context_window
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
        
    model.eval()
    
    # Get the model's device
    device = next(model.parameters()).device
    
    # Determine max length, ensuring it doesn't exceed context window
    context_window = getattr(model, 'context_window', 2)  # Default to 2 if not set
    max_length = max_length or max(1, context_window - len(input_str))

    # Convert input string to byte-level tokens
    input_tokens = torch.tensor([ord(c) for c in input_str], dtype=torch.long, device=device)
    
    # Pad or truncate to full context window
    if input_tokens.size(0) > context_window:
        input_tokens = input_tokens[-context_window:]
    
    # Pad with zeros if shorter than context window
    if input_tokens.size(0) < context_window:
        padding = torch.zeros(context_window - input_tokens.size(0), 
                            dtype=torch.long, 
                            device=device)
        input_tokens = torch.cat([padding, input_tokens])
    
    # Ensure batch dimension
    input_tokens = input_tokens.unsqueeze(0)
    
    if verbose:
        print("\n--- Generation Debugging ---")
        print(f"Input String: {input_str}")
        print(f"Input Tokens (decimal): {input_tokens.tolist()[0]}")
        print(f"Input Tokens (chars):   {[chr(t) for t in input_tokens.tolist()[0]]}")
        print(f"Context Window: {context_window}")
        print(f"Temperature: {temperature}")

    # Generate tokens autoregressively
    generated: List[int] = []
    with torch.no_grad():
        for step in range(max_length):
            # Get model predictions for entire input
            logits = model(input_tokens)
            
            # Focus on the last token's prediction
            next_token_logits = logits[0, -1, :]
            
            # Normalize logits
            next_token_logits = next_token_logits - next_token_logits.max()

            # Apply temperature scaling
            probs = torch.softmax(next_token_logits / temperature, dim=-1)

            if verbose:
                print(f"\nGeneration Step {step}:")
                print("Top 10 token probabilities:")
                top_probs, top_indices = torch.topk(probs, 10)
                for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    print(f"  Token {idx} ('{chr(idx) if 32 <= idx <= 126 else '?'}'): {p:.4f}")

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated.append(next_token.item())
            
            if verbose:
                print(f"  Selected Token: {next_token.item()} "
                      f"('{chr(next_token.item()) if 32 <= next_token.item() <= 126 else '?'}')")
            
            # Update input tokens, rolling the context window
            input_tokens = torch.cat([input_tokens[:, 1:], next_token.view(1, 1)], dim=1)

    # Convert tokens back to string
    generated_str = ''.join(chr(t) for t in generated)
    
    if verbose:
        print("\nGeneration Result:")
        print(f"Generated String: {generated_str}")
        print("--- End of Generation Debugging ---\n")

    return generated_str

def load_model(checkpoint_path: str) -> nn.Module:
    """
    Load a model from a checkpoint with strict parameter requirements.
    
    This function enforces that all required parameters must be present in
    the checkpoint configuration. It will not provide defaults and will
    fail if any parameter is missing.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        nn.Module: Loaded model in eval mode
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        KeyError: If any required parameter is missing from checkpoint
        RuntimeError: If checkpoint loading fails
    """
    from .model import FluctlightTransformer
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Get config, ensuring it exists
        if "state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing state_dict")
        if "config" not in checkpoint["state_dict"]:
            raise KeyError("Checkpoint missing configuration")
            
        config = checkpoint["state_dict"]["config"]
        
        # Required parameters - will raise KeyError if any are missing
        required_params = [
            "vocab_size",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "learning_rate",
            "weight_decay",
            "context_window",
            "v_scale"
        ]
        
        # Verify all required parameters exist
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise KeyError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Create model with exact parameters from checkpoint
        model = FluctlightTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            context_window=config["context_window"],
            v_scale=config["v_scale"]
        )
        
        # Load state dict
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k != "config"}
        model.load_state_dict(state_dict, strict=True)  # Changed to strict=True
        
        model.eval()
        return model
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}") from e
    except KeyError as e:
        raise KeyError(f"Invalid checkpoint format: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

def calculate_rmse(expected: str, actual: str) -> float:
    """
    Calculate Root Mean Square Error between expected and actual strings.
    
    For strings of different lengths, pads the shorter one with NUL bytes.
    Each character is treated as its ASCII value for the calculation.
    
    Args:
        expected: Expected string
        actual: Actual generated string
        
    Returns:
        float: RMSE value, or math.inf if strings are empty
    """
    if not expected or not actual:
        return math.inf
        
    # Pad shorter string with NUL bytes
    max_len = max(len(expected), len(actual))
    expected_padded = expected.ljust(max_len, '\0')
    actual_padded = actual.ljust(max_len, '\0')
    
    # Convert to ASCII values and calculate RMSE
    expected_vals = [ord(c) for c in expected_padded]
    actual_vals = [ord(c) for c in actual_padded]
    
    squared_diff_sum = sum((e - a) ** 2 for e, a in zip(expected_vals, actual_vals))
    rmse = math.sqrt(squared_diff_sum / max_len)
    
    return rmse