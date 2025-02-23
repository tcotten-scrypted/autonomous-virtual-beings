"""
Tests for utility functions.
"""

import pytest
import torch

from fluctlight.utils import decode_base64_pair, encode_base64_pair, generate_continuation
from fluctlight.model import MinimalTransformer

def test_base64_encoding_decoding():
    """Test Base64 encoding and decoding of pairs."""
    input_str = "Hello"
    output_str = "World"
    
    # Encode
    encoded = encode_base64_pair(input_str, output_str)
    
    # Decode
    decoded_input, decoded_output = decode_base64_pair(encoded)
    
    assert decoded_input == input_str
    assert decoded_output == output_str

def test_text_generation():
    """Test text generation utility."""
    # Create model
    model = MinimalTransformer()
    
    # Test generation
    input_str = "Hello"
    output = generate_continuation(
        model,
        input_str,
        max_length=10,
        temperature=0.8
    )
    
    assert isinstance(output, str)
    assert len(output) <= 10

def test_temperature_effect():
    """Test that temperature affects generation randomness."""
    model = MinimalTransformer()
    input_str = "Test"
    
    # Generate with high temperature (more random)
    high_temp_outputs = [
        generate_continuation(model, input_str, temperature=2.0)
        for _ in range(5)
    ]
    
    # Generate with low temperature (more deterministic)
    low_temp_outputs = [
        generate_continuation(model, input_str, temperature=0.1)
        for _ in range(5)
    ]
    
    # High temperature should give more varied outputs
    high_temp_unique = len(set(high_temp_outputs))
    low_temp_unique = len(set(low_temp_outputs))
    
    assert high_temp_unique >= low_temp_unique
