"""Tests for utility functions."""

import pytest
from fluctlight.utils import decode_base64_pair

def test_decode_base64_pair():
    """Test basic base64 decoding."""
    encoded = "aGVsbG8=\td2F2ZQ=="  # "hello\twave" in base64
    input_str, output_str = decode_base64_pair(encoded)
    assert input_str == b"hello"
    assert output_str == b"wave"

def test_decode_base64_pair_with_padding():
    """Test base64 decoding with different padding lengths."""
    encoded = "YQ==\tYmM="  # "a\tbc" in base64
    input_str, output_str = decode_base64_pair(encoded)
    assert input_str == b"a"
    assert output_str == b"bc"

def test_decode_base64_pair_invalid():
    """Test error handling for invalid base64."""
    with pytest.raises(ValueError):
        decode_base64_pair("invalid base64!")