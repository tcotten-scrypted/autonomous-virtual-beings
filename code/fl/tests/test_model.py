"""
Tests for the minimal Transformer model.
"""

import torch
import pytest

from fluctlight.model import MinimalTransformer

def test_model_initialization():
    """Test that model initializes correctly."""
    model = MinimalTransformer()
    
    assert model.vocab_size == 256
    assert model.d_model == 4
    assert model.n_heads == 2
    assert model.n_layers == 2
    assert model.head_dim == 2

def test_model_forward():
    """Test model forward pass."""
    model = MinimalTransformer()
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, 256, (batch_size, seq_len))
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, model.vocab_size)

def test_rope_application():
    """Test Rotary Positional Embedding."""
    model = MinimalTransformer()
    
    # Create sample queries and keys
    batch_size = 2
    seq_len = 8
    n_heads = 2
    head_dim = 2
    
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    
    # Apply RoPE
    q_rotated, k_rotated = model._apply_rope(q, k, seq_len)
    
    # Check shapes
    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape
    
    # Check that different positions have different rotations
    pos1 = q_rotated[0, 0, 0]  # First position
    pos2 = q_rotated[0, 0, 1]  # Second position
    assert not torch.allclose(pos1, pos2)

def test_causal_attention_mask():
    """Test that causal masking works correctly."""
    model = MinimalTransformer()
    
    # Create sample sequence
    batch_size = 1
    seq_len = 4
    x = torch.randint(0, 256, (batch_size, seq_len))
    
    # Get attention scores (extract from forward pass)
    with torch.no_grad():
        h = model.token_embed(x)
        layer = model.layers[0]
        
        q = layer["Wq"](h)
        k = layer["Wk"](h)
        v = layer["Wv"](h)
        
        q = q.view(batch_size, seq_len, model.n_heads, model.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, model.n_heads, model.head_dim).permute(0, 2, 1, 3)
        
        q, k = model._apply_rope(q, k, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (model.head_dim ** 0.5)
        
        # Check that future positions are masked
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        assert torch.all(attn_scores.masked_select(mask) == float('-inf'))
