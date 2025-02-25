"""
Tests for the minimal Transformer model.
"""

import torch
import pytest

from fluctlight.model import FluctlightTransformer

def test_model_initialization():
    """Test that model initializes correctly."""
    model = FluctlightTransformer()
    
    assert model.vocab_size == 256
    assert model.d_model == 4
    assert model.n_heads == 2
    assert model.n_layers == 2
    assert model.head_dim == 2

def test_model_forward():
    """Test model forward pass."""
    model = FluctlightTransformer()
    
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
    model = FluctlightTransformer()
    
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
    model = FluctlightTransformer()
    
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

def test_model_init():
    """Test model initialization with default parameters."""
    model = FluctlightTransformer(device=torch.device('cpu'))  # Force CPU
    assert model.vocab_size == 256
    assert model.d_model == 4
    assert model.n_heads == 2
    assert model.head_dim == 2
    assert len(model.layers) == 2

def test_context_window():
    """Test context window enforcement."""
    model = FluctlightTransformer(device=torch.device('cpu'))
    x = torch.randint(0, 256, (1, 70))
    output = model(x)
    assert output.shape[1] == 63  # MAX_CONTEXT - 1

def test_rope_application():
    """Test RoPE transformation."""
    model = FluctlightTransformer(device=torch.device('cpu'))
    q = torch.randn(1, 2, 4, 2)
    k = torch.randn(1, 2, 4, 2)
    v = torch.randn(1, 2, 4, 2)
    q_rope, k_rope, v_rope = model._apply_rope(q, k, v, 4)
    assert q_rope.shape == q.shape
    assert k_rope.shape == k.shape
    assert v_rope.shape == v.shape
    assert not torch.allclose(q, q_rope)
    assert not torch.allclose(k, k_rope)
    assert not torch.allclose(v, v_rope)

def test_forward_shape():
    """Test forward pass shapes."""
    model = FluctlightTransformer(device=torch.device('cpu'))
    x = torch.randint(0, 256, (2, 10))
    output = model(x)
    assert output.shape == (2, 10, 256)

def test_training_step():
    """Test training step with sequence shifting."""
    model = FluctlightTransformer(device=torch.device('cpu'))
    input_seq = torch.randint(0, 256, (2, 10))
    target_seq = torch.randint(0, 256, (2, 10))
    loss = model.training_step((input_seq, target_seq), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar loss
