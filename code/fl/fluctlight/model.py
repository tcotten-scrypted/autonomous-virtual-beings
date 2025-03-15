"""
Implementation of a Fluctlight Transformer model with Rotary Positional Embeddings (RoPE).

This implementation focuses on efficiency and clarity while maintaining core Transformer 
functionality. The model uses byte-level tokenization (vocab size: 256) and RoPE for
enhanced position-aware attention.

Key Architecture Points:
- Parameters: 2,656 (including final normalization layer)
- Vocabulary: 256 tokens (byte-level encoding)
- Embedding Dimension: 4 (compact but effective)
- Attention Heads: 2 (each head dimension: 2)
- Feed-forward Dimension: 8 (2x embedding dimension)
- Default Context Window: 2 tokens (minimum viable for pattern learning)
- Position Encoding: Rotary Positional Embedding (RoPE) with context-window scaling

The small dimensions were chosen to demonstrate Transformer concepts while remaining
computationally efficient. Despite its size, the model can effectively learn 
patterns in byte-level encoded text.

Empirical Evidence:
- Stable training with up to 16 active tokens from the 256-token vocabulary
- Successfully learns alternating patterns (e.g., "ababab") at low temperatures
- 2-token context window sufficient for basic pattern extrapolation
- RoPE scaling enables position-aware attention even in minimal context

Note on Positional Encoding: Standard RoPE parameters were designed for models with hundreds 
or thousands of dimensions (e.g., 10000 as the frequency base in GPT models). For our tiny model 
with only 4 dimensions total, these parameters would create extremely slow-varying positional 
signals, making nearby positions indistinguishable in the limited embedding space. 

Our implementation scales the frequency by the context window size, effectively compressing 
the full range of positional information into our small context. This ensures that each 
position within our context window receives a distinct encoding, even with the severe 
dimensional constraints. Without this scaling, our tiny model would struggle to differentiate 
between positions, limiting its ability to learn sequence-dependent patterns.
"""

import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, Callable

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

    @staticmethod
    def dropout_rate(d_model: int) -> float:
        """
        Calculate dynamic dropout rate based on model size.
        
        For tiny models (pre-Origami expansions), we want negligible dropout.
        For larger models (post-Origami expansions), we want more dropout,
        but capped at 10% to avoid over-dropping information.
        
        Args:
            d_model: Model dimension
            
        Returns:
            float: Dropout rate between 0.0 and 0.1
        """
        return min(0.1, 0.5 * (d_model / 256))

    def __init__(
        self, 
        vocab_size: int = 256,
        d_model: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        context_window: Optional[int] = None,
        v_scale: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Fluctlight Transformer.

        Args:
            vocab_size: Size of vocabulary (default: 256 for byte-level)
            d_model: Dimension of embeddings (default: 4)
            n_heads: Number of attention heads (default: 2)
            n_layers: Number of transformer layers (default: 2)
            d_ff: Feed-forward network dimension (default: 8)
            learning_rate: Learning rate for optimization (default: 1e-3)
            weight_decay: Weight decay for regularization (default: 1e-5)
            context_window: Size of context window (default: None, will be predicted)
            v_scale: Scale factor for RoPE on value vectors (default: 0.0)
            device: Device to place model on (default: None, auto-detect)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize context window with prediction if not provided
        predicted_context = self.predict_context_window()
        if context_window is not None:
            # If context window is provided, ensure it's at least 2
            self.context_window = max(2, context_window)
            if self.context_window < predicted_context:
                print(f"Warning: Provided context window ({self.context_window}) is smaller "
                      f"than predicted optimal size ({predicted_context})")
        else:
            self.context_window = predicted_context
        
        self.v_scale = v_scale
        
        # Compute dynamic dropout rate based on model size
        # Small models (pre-Origami expansions) should negligibly drop tokens
        # Large models (post-Origami expansions) should drop more tokens
        # but cap at 10% to avoid over-dropping
        self.dropout_rate = FluctlightTransformer.dropout_rate(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(d_model)

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
        Predicts the optimal context window for the transformer model.
        
        The context window is scaled based on model dimensions with a floor of 2 tokens.
        For the default tiny model (d_model=4, n_layers=2, n_heads=2), this ensures we
        start at exactly 2 tokens, which represents the smallest useful context that can
        demonstrate pattern learning and extrapolation.
        
        For larger models, the context window scales with:
        base_context = (d_model * n_layers) // n_heads
        
        This scaling ensures that:
        1. Tiny model (d_model=4, n_layers=2, n_heads=2) -> base_context = 4 -> final = 2
        2. Medium models scale proportionally with depth and width
        3. Large models maintain efficient scaling relative to parameter count
        
        Returns:
            int: The optimal context length, minimum of 2
        """
        # Calculate base context size using the scalable formula
        base_context = (self.d_model * self.n_layers) // self.n_heads
        
        # For the tiny model configuration, ensure we get exactly 2
        if self.d_model == 4 and self.n_layers == 2 and self.n_heads == 2:
            return 2
        
        # For all other configurations, use the scalable formula with minimum of 2
        context = max(2, base_context)
        
        # Suggest maximum based on model dimension
        suggested_max = self.d_model * 4
        if context > suggested_max:
            print(f"Warning: Large context window ({context}) may impact performance. "
                  f"Consider using {suggested_max} for better efficiency.")
        
        return context
    
    def calculate_adaptive_angle_rates(self, dim_idx):
        """
        Adaptive RoPE frequency calculation that works across all context window sizes.
        Addresses both tiny (2) and massive (100k) context windows with appropriate scaling.
        """
        # Normalize dimension index for consistent scaling
        normalized_dim = dim_idx / (self.d_model // 2)
        
        # Piecewise frequency scaling based on context window size
        if self.context_window < 16:
            # Tiny contexts (2-16): Steeper hyperbolic curve
            # Creates maximal distinction between positions in minimal space
            # Critical for d_model=4 to learn position-dependent patterns
            base = 2.0
            scale = max(1.0, 16.0 / self.context_window)
        elif self.context_window < 1024:
            # Standard contexts: Traditional transformer scaling
            # Well-tested approach for normal range contexts
            base = 10000.0
            scale = 1.0
        else:
            # Massive contexts (1k-100k): Logarithmic scaling
            # Prevents frequency collapse at extreme distances
            # Maintains positional sensitivity across huge contexts
            log_adjustment = math.log(self.context_window / 1024 + 1)
            base = 10000.0 * log_adjustment
            scale = 1.0 / math.sqrt(log_adjustment)
        
        # Apply the appropriate curve based on context size
        angle_rates = 1.0 / (base ** (normalized_dim * scale))
        
        return angle_rates
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        v_scale: float = 0.0  # Changed default to 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Positional Embedding (RoPE) to Q, K, and optionally V with scaling.
        
        RoPE is traditionally applied only to Q and K vectors. The v_scale parameter allows
        experimental application to V vectors, which can be useful for:
        1. Interpolation studies (0.0 to 1.0)
        2. Enhanced positional awareness in small models
        3. Testing position-dependent value transformations
        
        Args:
            q: Query tensor of shape [batch, seq_len, d_model]
            k: Key tensor of shape [batch, seq_len, d_model]
            v: Value tensor of shape [batch, seq_len, d_model]
            seq_len: Length of the sequence
            v_scale: How much RoPE to apply to value vectors (0.0 = none, 1.0 = full)
                    Default is 0.0 (disabled) for standard behavior
        
        Returns:
            Tuple of (rotated_q, rotated_k, optionally_rotated_v)
        """
        
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for correct attention splitting.")

        # Generate position-dependent rotation angles scaled by the context window
        pos = torch.arange(seq_len, device=self.device, dtype=torch.float).unsqueeze(1)
        
        dim_idx = torch.arange(self.head_dim // 2, device=self.device, dtype=torch.float)
        angle_rates = self.calculate_adaptive_angle_rates(dim_idx)
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
    
    def calculate_norm_scale_factor(self) -> float:
        """
        Calculate a scaling factor for layer normalization based on model size.
        
        This implements a smooth transition from no normalization at d_model=4
        to full normalization at d_model=32 and above. At the current default
        size (d_model=4), this returns 0.0, preserving the model's behavior
        exactly. As the model grows through Origami expansion, normalization
        is gradually introduced.
        
        The scaling follows:
        - d_model = 4: scale = 0.0 (no normalization)
        - d_model = 16: scale ≈ 0.43 (partial normalization)
        - d_model = 32: scale = 1.0 (full normalization)
        - d_model > 32: scale = 1.0 (capped)
        
        Returns:
            float: Normalization scale factor between 0.0 and 1.0
        """
        return min(1.0, max(0.0, (self.d_model - 4) / 28))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer."""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Calculate normalization scale factor
        norm_scale = self.calculate_norm_scale_factor()
        
        # Enforce context window limit
        B, seq_len = x.shape
        if seq_len < self.context_window:
            # Create padded tensor filled with NUL tokens ()
            padded_x = torch.zeros((B, self.context_window), 
                                    dtype=x.dtype, 
                                    device=x.device)
            padded_x[:, -seq_len:] = x
            x = padded_x
        else:
            x = x[:, -self.context_window:]  # Take last context_window tokens
        
        B, seq_len = x.shape

        # Token embedding
        h = self.token_embed(x)

        # Process through transformer layers
        for layer in self.layers:
            # Multi-head self-attention with RoPE
            raw_h = h  # Store original pre-normalized activations
            norm_h = layer["ln1"](h)
            # Apply scaled normalization
            attn_input = norm_h * norm_scale + raw_h * (1 - norm_scale)
            
            # Compute Q, K, V projections
            q = layer["Wq"](attn_input)
            k = layer["Wk"](attn_input)
            v = layer["Wv"](attn_input)

            # Reshape for attention heads
            q = q.view(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(B, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

            # Apply RoPE to Q and K with stored v_scale
            q, k, v = self._apply_rope(q, k, v, seq_len, self.v_scale)

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
            raw_h = h  # Store original pre-normalized activations
            norm_h = layer["ln2"](h)  # Second layer norm before FF
            # Apply scaled normalization
            ff_input = norm_h * norm_scale + raw_h * (1 - norm_scale)
            
            ff_out = layer["ff_out"](F.relu(layer["ff_in"](ff_input)))
            ff_out = self.dropout(ff_out)
            h = h + ff_out  # Residual connection
        
        # Final layer norm
        raw_h = h  # Store original pre-normalized activations
        norm_h = self.final_ln(h)
        # Apply scaled normalization
        h = norm_h * norm_scale + raw_h * (1 - norm_scale)

        # Final output projection
        logits = self.output_proj(h)
        
        return logits

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Move batch to correct device and handle format
        if isinstance(batch, (tuple, list)):
            input_seq, target_seq = batch
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
        else:
            raise ValueError("Expected tuple of (input, target)")
        
        # Forward pass
        logits = self(input_seq)  # shape: [B, seq_len, vocab_size]
        
        # Compute loss with increased label smoothing
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_seq.reshape(-1),
            label_smoothing=0.1
        )
        
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Move batch to correct device and handle format
        if isinstance(batch, (tuple, list)):
            input_seq, target_seq = batch
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
        else:
            raise ValueError("Expected tuple of (input, target)")
        
        # Forward pass
        logits = self(input_seq)  # shape: [B, seq_len, vocab_size]
        
        # Compute validation loss
        val_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_seq.reshape(-1),
            label_smoothing=0.0
        )
        
        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the model's state dictionary including configuration parameters.

        Returns:
            Dict containing model state and configuration
        """
        state = super().state_dict()
        # Add configuration parameters
        state["config"] = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "context_window": self.context_window,
            "v_scale": self.v_scale,
        }
        return state

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        **kwargs
    ) -> "FluctlightTransformer":
        """
        Load a pretrained model from a checkpoint.
        
        This custom implementation ensures proper handling of our config
        dictionary and parameter overrides.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            map_location: PyTorch device mapping
            **kwargs: Additional arguments to override saved config
            
        Returns:
            FluctlightTransformer: Loaded model instance
        """
        if map_location is None:
            map_location = get_default_device()
            
        # Load checkpoint with map_location
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract config, defaulting to empty dict if not found
        config = checkpoint.get("config", {})
        
        # Get stored context window with a default of None to trigger prediction
        stored_context = config.get("context_window", None)
        
        # Create new model instance with saved config
        model = cls(
            vocab_size=config.get("vocab_size", 256),
            d_model=config.get("d_model", 4),
            n_heads=config.get("n_heads", 2),
            n_layers=config.get("n_layers", 2),
            learning_rate=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
            context_window=stored_context,  # Let model predict if not stored
            v_scale=config.get("v_scale", 0.0),
            **kwargs  # Allow overriding any parameter
        )
        
        # Load the state dict, excluding our custom config
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k != "config"}
        model.load_state_dict(state_dict, strict=False)
        
        # If context window was stored, ensure it's used (in case prediction was larger)
        if stored_context is not None:
            model.context_window = max(2, stored_context)  # Maintain minimum of 2
        
        return model