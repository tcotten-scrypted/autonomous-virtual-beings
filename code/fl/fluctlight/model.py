"""
Implementation of a Fluctlight Transformer model with Rotary Positional Embeddings (RoPE).

This implementation focuses on efficiency and clarity while maintaining core Transformer 
functionality. The model uses byte-level tokenization (vocab size: 256) and RoPE for
enhanced position-aware attention.

Key Architecture Points:
- Vocabulary: 256 tokens (byte-level encoding)
- Embedding Dimension: 4 (compact but effective)
- Attention Heads: 2 (each head dimension: 2)
- Feed-forward Dimension: 8 (2x embedding dimension)
- Context Window: 16 tokens
- Position Encoding: Rotary Positional Embedding (RoPE)

The small dimensions were chosen to demonstrate Transformer concepts while remaining
computationally efficient. Despite its size, the model can effectively learn 
patterns in byte-level encoded text.
"""

import math
import numpy as np
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

class FluctlightTransformer(pl.LightningModule):
    """
    A Fluctlight Transformer implementation using Rotary Positional Embeddings (RoPE).
    """

    def __init__(
        self, 
        vocab_size: int = 256,
        d_model: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        context_window: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.context_window = context_window or self.predict_context_window()
        
        # Compute dynamic dropout rate based on model size
        # Small models (pre-Origami expansions) should neglibibly drop tokens
        # Large models (post-Origami expansions) should drop more tokens
        # but cap at 10% to avoid over-dropping
        self.dropout_rate = FluctlightTransformer.dropout_rate(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

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

        # Move model to detected device
        self.to(self._device)

        # Save hyperparameters
        self.save_hyperparameters(ignore=['device'])

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device
    
    def predict_context_window(self) -> int:
        """
        Predicts the minimal viable context window for the transformer model.

        This is only a rough estimate and may differ from calculations such as GPT-2,
        where we might estimate 3,072 tokens but they actually use 1,024.

        Returns:
            int: The estimated optimal context length.
        """
        
        return max(16, (self.d_model * self.n_layers) // self.n_heads)
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        v_scale: float = 0.0  # Parameter controlling how much RoPE to apply to V (0.0-1.0)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Positional Embedding (RoPE) to Q, K, and optionally V with scaling.
        
        Args:
            q, k, v: Query, key and value tensors
            seq_len: Sequence length
            v_scale: How much RoPE to apply to value vectors (0.0 = none, 1.0 = full)
        """
        
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for correct attention splitting.")

        # Generate position-dependent rotation angles
        pos = torch.arange(seq_len, device=self.device, dtype=torch.float).unsqueeze(1)
        dim_idx = torch.arange(self.head_dim // 2, device=self.device, dtype=torch.float)
        angle_rates = 1.0 / (10000 ** (2 * dim_idx / self.d_model))
        angles = pos * angle_rates
        
        # Prepare rotation matrices
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        
        def apply_rotary(tensor):
            """Applies RoPE transformation safely, checking dimensions."""
            if tensor.shape[-1] % 2 != 0:
                raise ValueError(f"RoPE requires an even last dimension, but got shape {tensor.shape}")
            
            even, odd = tensor[..., 0::2], tensor[..., 1::2]
            rotated_even = even * cos - odd * sin
            rotated_odd = even * sin + odd * cos
            return torch.cat([rotated_even, rotated_odd], dim=-1)
        
        # Apply RoPE to Q and K
        q, k = apply_rotary(q), apply_rotary(k)
        
        # Apply scaled RoPE to V if v_scale > 0
        if v_scale > 0:
            # Store original V
            v_original = v
            
            # Apply rotation to V
            v_rotated = apply_rotary(v)
            
            # Interpolate between original and rotated V based on scale
            v = v_scale * v_rotated + (1 - v_scale) * v_original
        
        return q, k, v
    
    # The dropout_rate uses a sigmoid function to smoothly transition from
    # d_min to d_max as d_model increases in size
    @staticmethod
    def dropout_rate(d_model, d_min=0.01, d_max=0.5, k=0.001, midpoint=512):
        return d_min + (d_max - d_min) / (1 + np.exp(-k * (d_model - midpoint)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer."""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Enforce context window limit
        available_context = self.context_window - 1
        if x.shape[1] > available_context:
            x = x[:, -available_context:]  # Take only the trailing tokens
        
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
            q, k, v = self._apply_rope(q, k, v, seq_len, 1.0)

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
            attn_out = self.dropout(attn_out)
            h = h + attn_out
            
            # Feed-forward with second layer norm
            ff_input = layer["ln2"](h)  # Second layer norm before FF
            ff_out = layer["ff_out"](F.relu(layer["ff_in"](ff_input)))
            ff_out = self.dropout(ff_out)
            h = h + ff_out  # Residual connection
        
        # Final layer norm
        h = nn.LayerNorm(self.d_model).to(self.device)(h)

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
            
        # Forward pass
        logits = self(input_seq)  # shape: [B, seq_len, vocab_size]
        
        # For next token prediction, we need to align:
        # - logits: remove last position since we don't have a target for it
        # - target_seq: remove first position since it's what we're conditioning on
        logits = logits[:, :-1, :]  # Remove last position prediction
        target_seq = target_seq[:, 1:]  # Remove first position target
        
        # Compute loss (flattening both tensors to 2D)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),  # Using reshape is slightly more efficient than view+contiguous
            target_seq.reshape(-1),  # Using reshape avoids potential contiguous issues
            label_smoothing=0.1
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
            
        # Forward pass
        logits = self(input_seq)  # shape: [B, seq_len, vocab_size]
        
        # For next token prediction alignment
        logits = logits[:, :-1, :]  # Remove last position prediction
        target_seq = target_seq[:, 1:]  # Remove first position target
        
        # Compute validation loss
        val_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_seq.reshape(-1),
            ignore_index=-100  # Added to match training (if needed)
        )
        
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)