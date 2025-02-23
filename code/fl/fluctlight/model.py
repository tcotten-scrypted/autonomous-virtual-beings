"""
Implementation of a minimal Transformer model with Rotary Positional Embeddings (RoPE).

This implementation focuses on efficiency and clarity while maintaining core Transformer 
functionality. The model uses byte-level tokenization (vocab size: 256) and RoPE for
enhanced position-aware attention.

Key Architecture Points:
- Vocabulary: 256 tokens (byte-level encoding)
- Embedding Dimension: 4 (compact but effective)
- Attention Heads: 2 (each head dimension: 2)
- Feed-forward Dimension: 8 (2x embedding dimension)
- Context Window: 64 tokens
- Position Encoding: Rotary Positional Embedding (RoPE)

The small dimensions were chosen to demonstrate Transformer concepts while remaining
computationally efficient. Despite its size, the model can effectively learn 
patterns in byte-level encoded text.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MinimalTransformer(pl.LightningModule):
    """
    A minimal Transformer implementation using Rotary Positional Embeddings (RoPE).

    The model processes byte-level tokens (0-255) through a compact Transformer 
    architecture. RoPE is applied to queries and keys in self-attention, providing
    relative positional information without separate positional embeddings.

    Args:
        vocab_size: Size of vocabulary (default: 256 for byte-level)
        d_model: Embedding dimension (default: 4)
        n_heads: Number of attention heads (default: 2)
        n_layers: Number of transformer layers (default: 2)
        d_ff: Feed-forward dimension (default: 8)
        learning_rate: Initial learning rate (default: 1e-3)
    """

    def __init__(
        self, 
        vocab_size: int = 256,
        d_model: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 8,
        learning_rate: float = 1e-3
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.learning_rate = learning_rate

        # Token embedding layer (byte-level tokens to d_model dimensions)
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Transformer layers with RoPE attention
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "Wq": nn.Linear(d_model, d_model),  # Query projection
                "Wk": nn.Linear(d_model, d_model),  # Key projection
                "Wv": nn.Linear(d_model, d_model),  # Value projection
                "Wo": nn.Linear(d_model, d_model),  # Output projection
                "ff_in": nn.Linear(d_model, d_ff),  # FF network in
                "ff_out": nn.Linear(d_ff, d_model), # FF network out
                "ln1": nn.LayerNorm(d_model),      # First layer norm
                "ln2": nn.LayerNorm(d_model)       # Second layer norm
            }) for _ in range(n_layers)
        ])

        # Output projection to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tab token ID (ASCII 9) used for sequence padding/masking
        self.tab_token_id = 9

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

    def _apply_rope(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Positional Embedding to query and key tensors.

        RoPE provides relative positional information by rotating vector pairs
        in q and k tensors based on their position in the sequence. This allows
        the attention mechanism to consider token positions without separate
        positional embeddings.

        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            seq_len: Length of input sequence

        Returns:
            Tuple of rotated (q, k) tensors with positional information
        """

        # Generate position-dependent rotation angles
        pos = torch.arange(seq_len, device=q.device, dtype=torch.float).unsqueeze(1)
        dim_idx = torch.arange(self.head_dim // 2, device=q.device, dtype=torch.float)
        angle_rates = 1.0 / (10000 ** (2 * dim_idx / self.d_model))
        angles = pos * angle_rates

        # Prepare rotation matrices
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim/2]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, L, head_dim/2]

        # Split into even and odd dimensions for rotation
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        # Apply rotation using sin/cos
        q_rotated_even = q_even * cos - q_odd * sin
        q_rotated_odd = q_even * sin + q_odd * cos
        k_rotated_even = k_even * cos - k_odd * sin
        k_rotated_odd = k_even * sin + k_odd * cos

        # Recombine rotated dimensions
        q = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).reshape(*q.shape)
        k = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).reshape(*k.shape)

        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        The input sequence is processed through:
        1. Token embedding
        2. Multiple transformer layers, each with:
           - Multi-head self-attention with RoPE
           - Feed-forward network
           - Layer normalization and residual connections
        3. Final output projection to vocabulary logits

        Args:
            x: Input tensor of token indices [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        B, seq_len = x.shape

        # Token embedding
        h = self.token_embed(x)  # [B, seq_len, d_model]

        # Process through transformer layers
        for layer in self.layers:
            # Multi-head self-attention with RoPE
            attn_input = layer["ln1"](h)

            # Compute Q, K, V projections
            q = layer["Wq"](attn_input)
            k = layer["Wk"](attn_input)
            v = layer["Wv"](attn_input)

            # Reshape for attention heads
            q = q.view(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # Apply RoPE to Q and K
            q, k = self._apply_rope(q, k, seq_len)

            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            # Compute attention weights and output
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

            # Reshape attention output and project
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, seq_len, self.d_model)
            attn_out = layer["Wo"](attn_out)

            # First residual connection and layer norm
            h = h + attn_out
            h = layer["ln1"](h)

            # Feed-forward network with ReLU
            ff_out = layer["ff_out"](F.relu(layer["ff_in"](h)))

            # Second residual connection and layer norm
            h = h + ff_out
            h = layer["ln2"](h)

        # Final output projection
        logits = self.output_proj(h)
        return logits

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step logic with masking for tab-delimited input.

        Handles sequence padding and masking using tab character (ASCII 9)
        for proper loss calculation.
        """
        # Handle batch format
        if isinstance(batch, (tuple, list)):
            seq, in_len = batch
        else:
            seq = batch

        # Prepare input and target (shifted by 1)
        inp = seq[:, :-1]
        target = seq[:, 1:]

        # Create ignore mask for input tokens before tab
        B, T = target.shape
        tab_positions = (seq == self.tab_token_id).int().argmax(dim=1)
        pos_idx = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        ignore_mask = pos_idx < tab_positions.unsqueeze(1)

        # Forward pass and compute masked loss
        logits = self(inp)
        target_masked = target.masked_fill(ignore_mask, -100)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_masked.reshape(-1),
            ignore_index=-100
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step with same masking logic as training.
        """
        # Similar to training step
        if isinstance(batch, (tuple, list)):
            seq, in_len = batch
        else:
            seq = batch

        inp = seq[:, :-1]
        target = seq[:, 1:]

        B, T = target.shape
        tab_positions = (seq == self.tab_token_id).int().argmax(dim=1)
        pos_idx = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        ignore_mask = pos_idx < tab_positions.unsqueeze(1)

        logits = self(inp)
        target_masked = target.masked_fill(ignore_mask, -100)
        val_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_masked.reshape(-1),
            ignore_index=-100
        )

        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizer with learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)