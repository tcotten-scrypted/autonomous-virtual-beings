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

def get_default_device() -> torch.device:
    """
    Get the optimal available device (Metal, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class MinimalTransformer(pl.LightningModule):
    """
    A minimal Transformer implementation using Rotary Positional Embeddings (RoPE).
    """

    def __init__(
        self, 
        vocab_size: int = 256,
        d_model: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 8,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.learning_rate = learning_rate

        # Set device (use passed device or detect optimal device)
        self._device = device if device is not None else get_default_device()

        # Token embedding layer
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "Wq": nn.Linear(d_model, d_model),
                "Wk": nn.Linear(d_model, d_model),
                "Wv": nn.Linear(d_model, d_model),
                "Wo": nn.Linear(d_model, d_model),
                "ff_in": nn.Linear(d_model, d_ff),
                "ff_out": nn.Linear(d_ff, d_model),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tab token ID (ASCII 9)
        self.tab_token_id = 9

        # Move model to detected device
        self.to(self._device)

        # Save hyperparameters
        self.save_hyperparameters(ignore=['device'])

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device

    def _apply_rope(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Rotary Positional Embedding."""
        # Generate position-dependent rotation angles
        pos = torch.arange(seq_len, device=self.device, dtype=torch.float).unsqueeze(1)
        dim_idx = torch.arange(self.head_dim // 2, device=self.device, dtype=torch.float)
        angle_rates = 1.0 / (10000 ** (2 * dim_idx / self.d_model))
        angles = pos * angle_rates

        # Prepare rotation matrices
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

        # Split into even and odd dimensions
        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        # Apply rotation
        q_rotated_even = q_even * cos - q_odd * sin
        q_rotated_odd = q_even * sin + q_odd * cos
        k_rotated_even = k_even * cos - k_odd * sin
        k_rotated_odd = k_even * sin + k_odd * cos

        # Recombine rotated dimensions
        q = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).reshape(*q.shape)
        k = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).reshape(*k.shape)

        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer."""
        # Ensure input is on correct device
        x = x.to(self.device)
        B, seq_len = x.shape

        # Token embedding
        h = self.token_embed(x)

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
                torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            # Compute attention weights and output
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

            # Reshape attention output and project
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, seq_len, self.d_model)
            attn_out = layer["Wo"](attn_out)

            # Residual connection and layer norm
            h = h + attn_out
            h = layer["ln1"](h)

            # Feed-forward network
            ff_out = layer["ff_out"](F.relu(layer["ff_in"](h)))

            # Second residual connection and layer norm
            h = h + ff_out
            h = layer["ln2"](h)

        # Final output projection
        logits = self.output_proj(h)
        return logits

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step with device handling."""
        # Move batch to correct device and handle format
        if isinstance(batch, (tuple, list)):
            input_seq, target_seq = batch  # Correctly unpack input and target
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
        else:
            raise ValueError("Expected tuple of (input, target)")

        # Forward pass and loss computation
        logits = self(input_seq)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_seq.view(-1),
            ignore_index=-100
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step with device handling."""
        # Move batch to correct device and handle format
        if isinstance(batch, (tuple, list)):
            input_seq, target_seq = batch  # Correctly unpack input and target
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
        else:
            raise ValueError("Expected tuple of (input, target)")

        # Forward pass and loss computation
        logits = self(input_seq)
        val_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_seq.view(-1),
            ignore_index=-100
        )

        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)