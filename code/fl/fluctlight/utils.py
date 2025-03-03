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
    temperature: float = 1.0,
    verbose: bool = False
) -> str:
    """
    Generate text continuation with detailed logging.
    """
    model.eval()
    
    # Determine max length, ensuring it doesn't exceed context window
    max_length = max_length or max(1, model.context_window - len(input_str))

    # Get the model's device
    device = next(model.parameters()).device

    # Convert input string to byte-level tokens
    input_tokens = torch.tensor([ord(c) for c in input_str], dtype=torch.long, device=device)
    
    # Pad or truncate to full context window
    context_window = model.context_window
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
    generated = []
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
                    print(f"  Token {idx} ('{chr(idx) if 32 <= idx <= 126 else '?'}': {p:.4f}")

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated.append(next_token.item())
            
            if verbose:
                print(f"  Selected Token: {next_token.item()} ('{chr(next_token.item()) if 32 <= next_token.item() <= 126 else '?'}')")
            
            # Update input tokens, rolling the context window
            input_tokens = torch.cat([input_tokens[:, 1:], next_token.view(1, 1)], dim=1)

    # Convert tokens back to string
    generated_str = ''.join(chr(t) for t in generated)
    
    if verbose:
        print("\nGeneration Result:")
        print(f"Generated String: {generated_str}")
        print("--- End of Generation Debugging ---\n")

    return generated_str