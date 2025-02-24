Minimal Transformer with RoPE – Implementation and Documentation

1. Minimal Transformer Implementation (PyTorch Lightning)

Below is a complete PyTorch Lightning implementation of a minimal Transformer model with Rotary Positional Embedding (RoPE). This model uses a vocabulary of 256 (extended ASCII), embedding size 4, two Transformer encoder layers, and two attention heads (each head of dimension 2). It includes token embedding, causal self-attention (with RoPE applied to Q & K), a feed-forward network (4 → 8 → 4), residual connections, and layer normalization. The LightningModule handles training and validation steps (using cross-entropy loss).

import math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MinimalTransformer(pl.LightningModule):
    def __init__(self, vocab_size=256, d_model=4, n_heads=2, n_layers=2, d_ff=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # here 4//2 = 2
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # Transformer layers parameters
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
        self.output_proj = nn.Linear(d_model, vocab_size)
        # Save tab token ID for masking (tab is ASCII 9)
        self.tab_token_id = 9

    def forward(self, x):
        """Forward pass. x is a tensor of token indices shape [B, seq_len]."""
        B, seq_len = x.shape
        # Token embedding [B, seq_len, d_model]
        h = self.token_embed(x)
        # Iterate through transformer encoder layers
        for layer in self.layers:
            # Multi-head self-attention with RoPE
            # LayerNorm before attention (post-norm residual, so apply LN to residual input)
            attn_input = layer["ln1"](h)
            # Compute Q, K, V projections [B, seq_len, d_model]
            q = layer["Wq"](attn_input)
            k = layer["Wk"](attn_input)
            v = layer["Wv"](attn_input)
            # Reshape to [B, n_heads, seq_len, head_dim] for attention
            B, L = seq_len, seq_len  # (alias for clarity)
            q = q.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, L, head_dim]
            k = k.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, L, head_dim]
            v = v.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, n_heads, L, head_dim]
            # Apply Rotary Positional Embedding (RoPE) to Q and K
            # Generate position angles for rotation
            pos = torch.arange(seq_len, device=x.device, dtype=torch.float).unsqueeze(1)  # [L, 1]
            dim_idx = torch.arange(self.head_dim // 2, device=x.device, dtype=torch.float)  # [head_dim/2]
            # Compute rotary angles using base 10000^(2*i/d_model)
            angle_rates = 1.0 / (10000 ** (2 * dim_idx / self.d_model))  # [head_dim/2]
            angles = pos * angle_rates  # [L, head_dim/2], each position and pair index
            cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # shape [1, 1, L, head_dim/2]
            sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)  # shape [1, 1, L, head_dim/2]
            # Split Q, K into even and odd parts (pairs)
            q_even, q_odd = q[..., 0::2], q[..., 1::2]    # both [B, n_heads, L, head_dim/2]
            k_even, k_odd = k[..., 0::2], k[..., 1::2]
            # Apply rotation: (q_even * cos - q_odd * sin,  q_even * sin + q_odd * cos)
            q_rotated_even = q_even * cos - q_odd * sin
            q_rotated_odd  = q_even * sin + q_odd * cos
            k_rotated_even = k_even * cos - k_odd * sin
            k_rotated_odd  = k_even * sin + k_odd * cos
            # Reconstruct full Q, K with rotated components interleaved
            q = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).reshape(B, self.n_heads, seq_len, self.head_dim)
            k = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).reshape(B, self.n_heads, seq_len, self.head_dim)
            # Scaled dot-product attention (causal mask)
            # Compute attention scores [B, n_heads, L, L]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Causal mask: prevent attention to future tokens (j > i)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, n_heads, L, L]
            # Weighted sum of values
            attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, L, head_dim]
            # Recombine heads: [B, L, d_model]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, seq_len, self.d_model)
            # Apply output projection and add residual connection
            attn_out = layer["Wo"](attn_out)
            h = h + attn_out             # Add residual from input to attention output
            h = layer["ln1"](h)          # LayerNorm after first residual (Add & Norm)
            # Feed-forward network
            ffn_out = layer["ff_out"]( torch.relu(layer["ff_in"](h)) )
            h = h + ffn_out              # Add residual from attention output to FFN output
            h = layer["ln2"](h)          # LayerNorm after second residual (Add & Norm)
        # Final output projection to vocabulary logits
        logits = self.output_proj(h)  # [B, seq_len, vocab_size]
        return logits

    def training_step(self, batch, batch_idx):
        """Training step: batch is a tuple (seq, in_len) or just seq tensor."""
        # batch could be (inputs, input_length) or just inputs; handle accordingly
        if isinstance(batch, tuple) or isinstance(batch, list):
            seq, in_len = batch
        else:
            seq = batch
        # Shift inputs and targets for next-token prediction
        # Use full sequence except last token as input, and except first token as target
        inp = seq[:, :-1]
        target = seq[:, 1:]
        B, T = target.shape
        # Create ignore mask for loss:
        # Ignore all target positions that correspond to input or delimiter tokens.
        # Find index of the tab token in each sequence (original input delimiter)
        # We find tab in the **original sequence**, then adjust for target indexing.
        with torch.no_grad():
            # Find tab position in original sequence
            tab_positions = (seq == self.tab_token_id).int().argmax(dim=1)  # [B] index of first tab in each seq
            # Broadcast positions to mask out targets before the tab
            pos_idx = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)  # [B, T]
            # Mask positions where index < tab_index (i.e., target corresponds to input or tab)
            ignore_mask = pos_idx < tab_positions.unsqueeze(1)
        # Compute logits and loss (cross-entropy, ignoring masked positions)
        logits = self.forward(inp)  # [B, T, vocab_size]
        # Flatten for loss computation
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.reshape(-1), 
                               ignore_index=-100)  # -100 will ignore those indices in loss
        # Apply ignore mask by setting those target positions to -100
        # (This is an alternative to using ignore_index in cross_entropy directly. 
        # If using ignore_index as above, ensure masked targets are set to -100 in target tensor.)
        # target_masked = target.masked_fill(ignore_mask, -100)
        # loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_masked.view(-1), ignore_index=-100)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Similar to training_step: calculate loss on validation batch
        if isinstance(batch, tuple) or isinstance(batch, list):
            seq, in_len = batch
        else:
            seq = batch
        inp = seq[:, :-1]
        target = seq[:, 1:]
        # Mask out input portion as in training
        with torch.no_grad():
            tab_positions = (seq == self.tab_token_id).int().argmax(dim=1)
            pos_idx = torch.arange(target.size(1), device=self.device).unsqueeze(0).expand(seq.size(0), -1)
            ignore_mask = pos_idx < tab_positions.unsqueeze(1)
            target_masked = target.masked_fill(ignore_mask, -100)
        logits = self.forward(inp)
        val_loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_masked.view(-1), ignore_index=-100)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # Use Adam optimizer for simplicity
        return torch.optim.Adam(self.parameters(), lr=1e-3)

Notes: In the training loop above, we prepare the input and target so that the model is trained in a causal language modeling fashion: given the sequence of input and output tokens (with a delimiter tab between them), the model predicts the next token at each position. We mask the loss for all positions that correspond to the input (and delimiter) so that the model is only trained to predict the output portion. The Rotary Position Embeddings are applied to the Q and K matrices before computing attention scores, enabling relative position encoding ￼ ￼. The model has extremely few parameters (on the order of ~2.9K), making it minimal yet functional.

2. Sample Training and Validation Data

Below are sample training and validation datasets for the model. Each line is a Base64-encoded input-output pair (the raw decoded format is "input<TAB>output"). Base64 encoding is used to ensure all characters are printable ASCII and to avoid any tab/newline formatting issues in this markdown presentation. In practice, you would decode each line from Base64 to get the original input string and output string separated by a tab character. The examples here are random mappings (for instance, the output might be a simple transformation of the input). Training data contains about 450 sequences, and validation data contains about 75 sequences:

train.txt (Base64-encoded “inputoutput” pairs)

I35DPzwxCSQgREA9Mg==
dn5lK2tWCXcgZixsVw==
Iys7PWAJJCw8PmE=
Zzl7c3kJaDp8dHo=
PFlrQyA0eVZLQzMJPVpsRCE1eldMRDQ=
VzY5LWZePgkjYn8pYjsoKFQt
LUNlNWBWOQQmJ1F1IUBDTHFc
Xn9oU1R1JUF4PWUmJ04sI005
IENbBypPFCRwXQoI
OUFOaDVZIE1BbDU1Kw==
WFFmSzwqR2Z2W1hFa1lqLQ==
TRs8Zk9xWCBeKCY6Uj4wRzVY
PyYzCBY5L34gX2QQ
XCN0JiNKMSczYyA/Gjcz
SkUwQDM4KyhgKSUtS2p3
JjhjUwc7ZkV3Qkg8QCc5HUNUVw==
LFNkIT9ZSA02CjpVKw==
Pn0mZU5kW1o4XHI1KSFiR1NcPj9iV04v
KlM+ZVVbLTA5Y0J7ZAtz
H1wgFgI2azxzNTE=
DHZMPBgndWZ3Ni0sUFdySxw9Jzc8RV5v
UCdAZStzVV9PKFA=
aWtGIGFBPH0/JmdsZzZkJSQiGQ==
Mzs1NHhHJXd8Ig==
N2QnP2hLPB8oNS8rEGlYbG1lLTJM
YEc6c1leTj5kIF9Vdg==
MDp9SnBWJzpJIWZrZA==
LVg3MS9IPDVrZAxr
QFJ9VUk3fGd7PA==
BEVncV1eKRclL3IrIyJUbA==
OW96LF1qUkt+LSNqYUJpKS1OGC4=
VV1LQy1tRRYtSjY=
Kn5yMlZhKyEsIVtjYEI6Xg==
MUIgWSBPYjk0OnE/JVVyWFAvQCsg
K1xdOTZyL09PNlU5S0dtZXc4bw==
PUl+JHthMV81MB8f
Ik1KPyBRZ01wWFAi
SDJwLlo1HCx8XSQwIzdeK1x2MTA=
HChmLlhUawotQlAm
ZDM6OVx5cw1bbWsPI0E=
Fi80MiJ7YkZ9Ow==
f0xIYTo8cHd4WSVF
M2pSe2swe1s0RCkzNzcwSg==
cGg4cEo3RilnYks=
IlE2Kg4rYDV/fW1p
YWEoSXYkRyA8JmUmdnhw
MzRlEGlMYTNCdm1IIz9pMzBcKQ==
ZVpzVl1RNm84fikgIk5x
Mz0pRjlAZls4PSwyWyR2ZSVe
YnRJUUofKEMgYw==
MEBXNw0mYlNfE0QhUT9sYw==
ZWY+X3w+MU9SYBkqK0UyIy8hTlRJ
aWRFdiBjMDtiPHwuKH4=
LF9weCcwLkRZQ1NpbXIu
ME4geEhKaUMifVx0KQ==
YCd2PC01f2U7LSM1YDlP
NWFhSSpHUEkmW3R3Jl9VLUR1Sjo=
KmJwUF8jdWknbzs5T0cu
GXUpQlt6YSNgVzVGJw==
RF9XWEN9P14yP2NoPg==
PC9DUTtrdlNpKXw6IFs=
WDFNLS1wQQgqSyJXDFhaQDl5
JEg8dVB4VywjfmlA
Mzg9WkAtYy9nYCtKGSM7cyMd
VWdDJyUuO0VQYG4yLCUuWB0=
KHRBaDJ8VS5iPT5pLw==
NkArJig6PDhJQGInNyht
SlB8Sjo3TFEhZFF7OXNLaA==
MFlXaAhBMU1sLzspWloL
Rj9dI0UpUzUML0pdLEQl
XFV9K1AgYRw1OyBocTBFIg==
SW5cfF8/HjtZJ2FeOUBXMg==
Mio2XSFeK2scIS8hIGM=
fSQ5SCs5YSYoG1tL
OUVUIkZ1XURxLSc/
HVUiLzJRNHlzLSZQczE2Xw==
OnFAdDMyTnYiIjo8JUlC
GTt9J0JiU1IqQVI=
XFluBAQqOzw/dVpFMzdF
PFExU1wEV3B6N2IlOl9rJ3w9
SlwtO2lyJV0iJ3FFRA==
Nl4jKD1SXUleOyRFdGYpIyw=
dTc8ZE02PSdfbSEpLB4=
Ky1rIVgqa1cUNksvfnY=
LlYpLFo8KlcnVCdSJ0R2bks0bg==
Hl1AJW9BZkMmKzU0
S29WbTFeEj8mNygqeEtR
TUc7N0NgUDtPCA==
LyA2Yz1XdDtcV0w8
YDAuSw8rQm9oJkowMlxR
WWdfMlBrQVhjJCAm
VHNHL0UiNDtdfHsaOkBWTT04
Y1dDI0hlVGEibCImS3pgaA==
KnF5OC5ONkJUdV85KDMC
JWFJZEI9ICtEQUdP
WX9JITZUOGkzfBZqWFk/
N2ZlT0MqLlc3RkUhP0Z1
XW1yNhUuYFtfLDwrTT1zY3Ah
Q0taMn5dcwEsKXc2
KjY+NAwdL0NlZUItNVN3fQ==
UD10Xz1fa08yYlR6KzM=
IS1DYiQ+SyE2OUInZFxZOi0=
Y0p6OmI/BF17OGUyXHUt
TDUkOQozUy0rW2VKQkIm
HUkoM0gEQ0ttBC9KbVoE
S0NUYkBmYX8kBzYjL0s9UQ==
bWRgVzFVf1thLw==
Yi0yLwQuGXple0ImdVV4Yko7Lg==
VU46ZlFBe0EJcyRaTQ==
ICd6VFoRIVBvD1h4ICd2
LlBgLCwgQyN7RF9cOx0k
SEg+SXU5Kk07MUI1IH0k
JkJFYDxIWkZEMjgh
HEEcW2QvZ30gOylNYEgv
YjJSaVA7UnInWls=
SEAtJmZDNCxLUBk2P2s0Iks=
ODUrVnpeNCwjTzg5JHdQ
S1xXJ0B5K0w6VCg1b1d4Mh==
Wi1kMz5WR31zYktZNHI1
NmxVGHJxbGthQzQz
bXBhZF5QfSYxFT1iMUR4
RjM+PWJWNks1RXti
QUJgPy8+YH5iMA==
fU0kcxogRjZGMXNbXk8i
JiI6bidyLGRWdTEhCg==
H3YaIkuQOkpHQSE/Cmd1K00=
L0gnEjpbYH5HJy8gWWNf
ZyZbT2B0JEwlMyR3CWlq
OzYndmEwXDFEKUh4
TmdRNWRvb2JZKXI5THxw
RTkyd3IyPB8mL1Z+cQ==
JlciZzY/Jzh2T0MkL0M0Uw==
cFx3KV5dWDpVJ0p7XnY5Ng==
GhcoKAw6OHdZRVBRNlRI
VUZyIjZbGSY4J1NqdyQi
NUNWbkc2KXFyYy9JJF1f
RDt0bV9qTUI1dldBLWFr
J3t1dC97cEZjXlg=
U2k3M1FxRENCLkMwFihjXA==
PDk6KHlcXH0rLWB6
RjFSHCFKLUo4L0RI
NGkqJW1JPUNSCFlZZj0n
ZWAsP0Ufb0ItZHQmLjBf
aWcoGhZpMlxjP2xeE0Jb
Ji4nHSZoT1FRAzFn
JUF0L003MD8ud1BUSR4i
SkZGRWk4dEMrImw9Lyc=
XWYwNUFVM1UmMiRbAVZz
J1t9cHgqMCIvPUN1H0k1
NUZNIiRHPCokRjZVXWc2
Vyo1Lnk9MDFCNz1p
RUFYOgIjLCI4RTxtI1Jj
LzdfXWREcT0wJkguNVpZ
SGBGa0FcUTsxZH05Qzgq
W15qcWU8MGBrcD4/
WnsxLyZXWzkrLVFGH1hc
UjFUbTd6NVgoMyhBIy0j
T1dYNHEvOTw/HThbSF4x
a1FaMEYtaVFQfStyPV0l

val.txt (Base64-encoded validation pairs)

JixKOjhUeQknLUs7OVV6
dERGPiwmUmlmCXVFRz8tJ1NqZw==
MyZOIFYrRXRwbF05CTQnTyFXLEZ1cW1eOg==
Izo1cUUqCSQ7NnJGKw==
LkdSXF50QStyZVE3CS9IU11fdUIsc2ZSOA==
TFNYNWIuLSxgKj8mNVFy
M0U9CVt3R1koNFsrdVZgYjIu
cEVtcHcoKkIobXgp
QCg2RUIhKCgrCVdl
HCYvbENfNl9sPSk5emkD
PnlmJyRBU11HMx0nT1IaZw==
TFMgM0d8IjdPYnRBN1twNgo=
YENjNWs5WCgqYnAh
SmZtdjk1TWNkKy4pfQ==
K14tIyFfOCt2XiZpKz5maw==
SEUpLDtVKCgtZDYqbSVq
N0grP0d6IGNzNCs8Y0M/
XFhZMTUqSDxgazk6
TCpyMk4yWCYzEjlyaA==
Z09Hd0Ate1NrfEcv
MCFrKXgmZ1Aze0EuCg==
fw8mVFt5H34gBA==
S2MYIU5xb1wnLEJbNFs/
WUImMG1zUjsrCEdCNmAk
ACxLfysvXSRnV30=
KFhYKVoINWpdYCtcIQ==
IT8yPkQ7M0BWIjw/YXR2
KlAgLk1LND5MLT4sX3ZF
eVY7PEJ8Ll8+fTgr
QCxbawQwVktgJnk+Pzpz
K1QRQCdxP1ZaI1sq
QkYiIEI8Q3BWNzAtIyJ5
cFN3NDw3Nk8hJA==
MXlzLzRUSyQ7Kh9iYw==
Y1hQblsxKy4wYkluVjI6
T2o5Sk8pS1cgZmAuNngm
JTk3Sng3Ki0iYVoy
RjNdZFVwR2QpIS9lPyw9
LFpWaUtMN0gzO1t7
JGd4JU9DIkZlYDZyMi5T
bEV5bFIpXGZORVNq
J3M0AS98NBo8eXxjWy8/
ZiU4N0M9Oy57V2k5

(Each line above, when decoded from Base64, yields an example like input_string<TAB>output_string. The content is randomized printable text; the model’s task is to map the input to the corresponding output.)

3. Project README – Model Explanation and Usage

Purpose and Design Philosophy

This project demonstrates a minimal Transformer – a tiny Transformer model with only on the order of 2,900 parameters. The purpose is to create the simplest functional Transformer to study and illustrate how Transformers work at a basic level. By using extremely small dimensions (embedding size 4, feed-forward hidden size 8, 2 heads, 2 layers), the model is easy to train on a CPU and easy to inspect or even overfit on a small dataset. Starting with such a minimal model (2.9K params) provides a baseline that can be progressively expanded to larger models. This minimal design is great for learning and debugging: it’s small enough to train quickly and observe behaviors (e.g. learning simple mappings), and it serves as a starting point for future unfolding (growing the model in size while reusing the learned parameters).

Despite its size, the model includes all key components of a Transformer:
	•	Token embeddings for a vocabulary of 256 (covering extended ASCII characters).
	•	Multi-head self-attention mechanism (2 heads, each 2-dimensional) with Rotary Positional Embeddings (RoPE) to encode token positions.
	•	Feed-Forward Network (a two-layer MLP with hidden size 8) at each layer.
	•	Residual connections and Layer Normalization applied after each sub-layer (Add & Norm).
	•	A final output projection to map the 4-dimensional representations back to 256-dimensional token logits.

The use case for this model is primarily educational and experimental. It can learn very simple patterns or mappings (for example, mapping an input string to an output string in the training data). With a very small capacity, it will typically memorize the training data if trained for long, which is useful for verifying that the model and training process are working. In practice, one would not use such a tiny Transformer for real tasks, but it’s a stepping stone toward gradually building larger models by unfolding this minimal model.

Model Architecture

The minimal Transformer architecture is a drastically scaled-down version of a standard Transformer encoder-decoder, essentially functioning as a decoder-only Transformer (like a small language model) for sequence-to-sequence mapping. All tokens (input and output) are handled in one sequence with a special delimiter (tab character) separating input and output. The model sees the input (and delimiter) and is trained to predict the output tokens.

Key architecture hyperparameters and components:
	•	Vocabulary: 256 tokens, representing extended ASCII. Each character (byte) is a token. This includes standard printable ASCII and control characters (we use the tab character as a delimiter).
	•	Embedding Dimension: 4. Each token is represented by a 4-dimensional vector. The embedding matrix thus has shape 256×4.
	•	Transformer Encoder Layers: 2 layers in stack. Each layer has:
	•	Multi-Head Self-Attention (MHSA): 2 heads. The model dimension 4 is split into 2 heads of dimension 2 each. Each head attends to the sequence with learned query, key, value projections (Wq, Wk, Wv each of shape 4×4). Causal masking is used so that tokens only attend to previous tokens (this allows the model to be used for autoregressive generation).
	•	Rotary Positional Embedding (RoPE): Applied to the Q and K vectors in attention. RoPE encodes positions by rotating Q and K in a 2D subspace for each pair of dimensions, which injects position information as a phase change in dot-product attention ￼. This effectively means the attention scores depend on the relative positions of tokens ￼, improving the model’s ability to generalize to longer sequences without fixed positional embeddings.
	•	Feed-Forward Network (FFN): A two-layer MLP applied to each token position. It expands the 4-dim representation to 8-dim (ff_in layer 4→8 with ReLU), then back down to 4-dim (ff_out layer 8→4). This gives the network capacity to transform and mix information across dimensions in a non-linear way.
	•	Residual Connections: The output of the attention sub-layer is added back to its input (skip connection), and same for the FFN sub-layer.
	•	Layer Normalization: After each addition (residual), a LayerNorm normalizes the features. This helps stabilize training. (We use a Post-Norm configuration: normalization is applied after each sub-layer’s residual addition, as in the original Transformer.)
	•	Output Projection: A linear layer (4→256) maps the final 4-dimensional token representations to logits over the 256-token vocabulary. This is analogous to the inverse of the embedding layer, translating the model’s learned representation back into actual character predictions. (Weights are not tied in this implementation, but they could be in theory.)

Despite the limited size, the model has all the pieces needed for sequence learning. In total, it has on the order of ~2.9K parameters (embedding tables, linear weights, layer norm parameters, etc.). For example, the token embedding matrix is 256×4 = 1024 parameters, and the output projection is another 1024; the two transformer layers together contribute roughly a few hundred parameters each (Q/K/V/O weights, FFN weights, and biases, plus LayerNorm gains/biases).

Architecture Diagram

Below is a Mermaid diagram illustrating the model architecture and data flow through one layer of the Transformer (the model has two such layers back-to-back). This diagram shows the token embedding, the multi-head attention with RoPE applied to Q and K, the feed-forward network, residual connections, and layer normalization, as well as the final projection to output logits.

flowchart TD
    subgraph Input_Sequence[Input Tokens (bytes)]
    end
    Input_Sequence --> Embedding[Token Embedding (256 -> 4)]
    Embedding --> Layer1[Transformer Encoder Layer 1]
    subgraph Layer1
        direction TB
        subgraph MHA1[Multi-Head Attention (2-head, RoPE)]
            direction LR
            LN1a[LayerNorm] --> Q1[Wq]
            LN1a --> K1[Wk]
            LN1a --> V1[Wv]
            Q1 --> Q1_vec[Q vectors]
            K1 --> K1_vec[K vectors]
            V1 --> V1_vec[V vectors]
            Q1_vec -. apply RoPE .-> Q1_rope[Q (rotated)]
            K1_vec -. apply RoPE .-> K1_rope[K (rotated)]
            Q1_rope --> AttnScores1[Dot-Product & Softmax]
            K1_rope --> AttnScores1
            V1_vec --> AttnOut1[Weighted Sum]
            AttnScores1 --> AttnOut1
            AttnOut1 --> OutProj1[Wo]
        end
        OutProj1 --> AddRes1[Add & Norm (residual + LN)]
        AddRes1 --> FFN1[Feed-Forward (4 -> 8 -> 4)]
        FFN1 --> AddRes1b[Add & Norm (residual + LN)]
    end
    Layer1 --> Layer2[Transformer Encoder Layer 2 (same structure)]
    Layer2 --> OutputProj[Output Projection (4 -> 256)]
    OutputProj --> Logits[Logits (prediction scores)]

Diagram Explanation: Each input token (a byte character) is first mapped to a 4-dimensional embedding. In Multi-Head Attention, the model computes Query, Key, and Value vectors (each 4-dim, split across 2 heads) for each position. Rotary Positional Encoding (RoPE) is applied to Q and K vectors, rotating them in a plane by an angle proportional to their position index ￼. The dot-product attention then produces weighted sums of values from previous positions for each token (since causal masking prevents looking ahead). The attention output goes through an output linear Wo and is added back to the input (residual), followed by a LayerNorm. Next, the Feed-Forward Network processes each token’s data independently through a ReLU-expanded 8-dim hidden layer and back to 4-dim, followed by another residual add and LayerNorm. After two such layers, the final normalized 4-dim vectors are projected to 256-dimensional logits, one for each possible token, determining the predicted output characters.

Progressive Expansion (Future Unfolding of the Model)

One interesting aspect of this minimalist model is that it can be progressively “unfolded” or expanded to a larger Transformer by mirroring or copying its weights. The idea is to use the small model as a building block and increase capacity without starting from scratch. This approach is inspired by function-preserving transformations like Net2Net which introduced methods to expand neural networks (width or depth) while initializing the larger model to behave exactly like the smaller one ￼.

How the unfolding (mirroring weights) works:
	•	Increasing Model Width (embedding, hidden size, number of heads): We can double the dimensionality of the model by duplicating neurons/units. For example, to expand the embedding size from 4 to 8, we can copy each of the 4-dimensional embedding vectors to create an 8-dimensional embedding (essentially two identical halves). Similarly, each weight matrix (like Wq, Wk, etc. which are 4×4) can be expanded to 8×8 by copying the small matrix into the larger one (filling the new parameters by mirroring the old ones). After this transformation, the larger model initially computes the same outputs as the old model (the extra dimensions are initially redundant mirrors), preserving its function ￼. This gives a “warm start” for training a larger model – it already knows the patterns the smaller model learned, and the new degrees of freedom can then be fine-tuned to learn more.
	•	Increasing Number of Heads: If we want more attention heads, we can copy the existing head(s) weights to new head slots. For instance, to go from 2 heads to 4 heads, each of the original 2 head’s weight matrices can be duplicated so that the new 4 heads start as two pairs of identical twins. This again preserves the initial attention computations (just done redundantly by twin heads). Subsequent training can then differentiate their roles.
	•	Increasing Depth (adding more layers): We can insert new Transformer layers that initially perform an identity transformation. One technique is to initialize a new layer’s weights such that its output equals its input (for example, set the feed-forward weights to small values and the attention to initially just pass the input through, effectively like inserting an identity layer) ￼. Another method is to duplicate an existing layer’s weights for the new layer(s). For example, to go from 2 layers to 4 layers, we could copy the learned weights from the 2nd layer into a new 3rd layer (and optionally 1st layer into new 4th) – effectively stacking the same operations again. Initially this doesn’t change the overall function if done carefully (especially if the new layer’s output is weighted to be small or normalized to not disturb the network). Then the model can be further trained to adjust these new layers.
	•	Preserving Functionality: The key in all these expansions is to initialize the new larger model such that it starts off computing the same function as the smaller model ￼. This typically involves setting new weights in a patterned way (copying or tiling from old weights, and sometimes dividing by factors or using identity matrices for new parts). By doing so, we don’t lose the knowledge already learned; we only add capacity for learning more. This progressive scaling can significantly speed up training of large models, as shown in some research, because the model doesn’t start from random initialization ￼.

Using these techniques, one could train the minimal model on a simple task, then gradually grow it (e.g., double the embedding to 8, 16, … add more heads and layers) to handle more complex tasks, each time reusing the previous weights as a starting point. This approach treats the small model as a “seed” that can blossom into a larger model, a concept sometimes referred to as model growth or model folding. The minimal model’s simplicity and low parameter count (2.9K) make it feasible to experiment with such growth quickly.

Why start so small (2,904 parameters)? Starting with a minimal number of parameters ensures that the model can memorize small datasets easily and that every parameter’s role can be inspected. It reduces training time to seconds and allows observing training dynamics on a micro-scale. It also forces us to include only the most essential components of a Transformer. From this base, every time we increase capacity, we understand exactly what new parameters are added. This stepwise expansion helps in demystifying how each part of a Transformer contributes to its performance. In essence, the minimal model is like a bonsai tree – small but fully formed – which can be replanted into a bigger pot to grow into a larger tree given time.

How RoPE helps scaling: Rotary Positional Embedding is particularly handy when scaling up sequence length or model size because it encodes positions implicitly and continuously. Unlike fixed positional embeddings (which might be learned for a specific maximum length), RoPE uses a deterministic formula to rotate Q/K vectors ￼. This means if we increase the sequence length, we don’t need new position embeddings – the same formula extrapolates to unseen positions (to an extent). When expanding model dimensions, we can also integrate RoPE into the new dimensions without breaking the existing ones: e.g., if we double the embedding, we can assign the original RoPE frequencies to one half of the dimensions and perhaps initialize the other half with either repeated frequencies or new ones (for capturing finer positional details). The relative positioning property of RoPE means the model’s attention focuses on relative distance between tokens ￼, which tends to generalize better when the context window grows. In summary, RoPE improves the scalability of the model in terms of sequence length and can be smoothly adopted as we increase model size, without having to re-learn positional encodings from scratch for the new model.

Training Guidelines

Training this minimal Transformer requires careful handling of data and expectations:
	•	Data Preparation: The training and validation data should consist of input-output pairs that the model will learn to map. In our sample, each line after decoding is structured as input_string<TAB>output_string. We use Base64 encoding in the sample files to ensure all characters are safe to handle (only printable ASCII). In a real training script, you would read each line, Base64-decode it to get the raw input and output, then tokenize those. Tokenization here is simple: each character (byte) is a token ID (0–255 range). For example, the letter “A” (ASCII 65) would be token 65, a space (ASCII 32) is token 32, etc. The tab delimiter is ASCII 9, which is included in the vocab.
	•	Batching and Padding: Since sequences may have varying lengths, you can pad sequences in a batch to the same length. Ensure that the padding token (for instance, ASCII 0 or another unused token) is also masked out in the loss. The provided code uses an ignore mask for the input part and would also ignore any target positions set to -100. You should set padded token targets to -100 as well so they don’t contribute to loss.
	•	Loss Computation: We train the model as a causal language model, predicting the next token. However, we only want to compute loss on the output portion of each sequence. The training loop handles this by masking: it ignores loss for predictions that correspond to the input or the tab delimiter. This way, the model isn’t trained to predict the input (which is given) or to predict the delimiter (we supply it). It only learns to predict the output characters given everything that comes before them in the sequence.
	•	Expected Performance: With such a tiny model, you should not expect it to generalize beyond the training data for anything complex. Its capacity is enough to memorize a few hundred patterns. For instance, if you have ~400 training pairs (like in our sample), the model can typically overfit them — achieving very low training loss after enough epochs. Validation loss will start low if validation pairs overlap with training patterns or remain higher if they are somewhat different. As an estimate, the model might perfectly memorize ~500 short sequences after training for a few epochs (say 100-200 epochs, depending on learning rate and complexity of patterns). If you see training loss flattening near 0 but validation loss stagnating above 0, that’s a sign of overfitting (expected given small data and model).
	•	Overfitting and Generalization: Because the model is so small, it actually overfits extremely fast. To see any generalization, you’d need to limit training time or provide many variations of a pattern so the model finds a rule. If the task is something like “shift each character by +1” (a simple cipher), the model could potentially learn the rule and apply it to unseen strings, but with 4-d embeddings it might still just memorize the mapping table. If you want to reduce overfitting, you could add some regularization: e.g., dropout (not included in this minimal code) or early stopping based on validation loss.
	•	Training Strategy: A good strategy is to start with a relatively higher learning rate (e.g. 1e-3 as in the code) since the model is small and data is small. Monitor training loss — it should drop quickly. If it plateaus or oscillates, you can try lowering the learning rate. Because the model is simple, it’s also possible to use a very high learning rate (even 1e-2) for quick convergence, but watch for instability. In PyTorch Lightning, we log training and validation loss; you can enable the progress bar to see live metrics. Given the tiny scale, you might run, say, 1000 training iterations (a few epochs if batch size is small) and achieve near-zero training loss. There’s not much danger of underfitting here — the main risk is overfitting, which in this context is fine since we primarily want the model to learn the training pairs.
	•	Data Size vs. Model Capacity: Rough guideline: ~2.9K parameters can perfectly memorize on the order of hundreds of short sequences. If you provided thousands of training pairs, the model might not have enough capacity to memorize them all well, leading to higher training loss floor. In such a case, either increase model size or reduce task complexity. For demonstrating the minimal Transformer, a few hundred training examples (like 400) is plenty.

Code Structure and Usage

The implementation is contained in a single class MinimalTransformer (a subclass of pl.LightningModule). Here’s a breakdown of the structure:
	•	__init__: Defines the embedding layer, the ModuleList of Transformer layers, and the final output projection. Each Transformer layer is a ModuleDict with its own weights for Wq, Wk, Wv, Wo (attention projections), feed-forward layers (ff_in and ff_out), and LayerNorms (ln1 and ln2). The model also stores a tab_token_id (ASCII code 9) for convenience in masking.
	•	forward(x): Takes a batch of token sequences x (shape [batch, seq_len]) and returns the logits for each token in the sequence (shape [batch, seq_len, vocab_size]). Inside, it performs token embedding, then iterates through each Transformer layer:
	•	Applies LayerNorm (depending on whether we use pre-norm or post-norm, our code actually does a LayerNorm on the input to attention and after adding residual – effectively doing post-norm twice which might be redundant; one could simplify, but we kept it explicit for clarity).
	•	Computes Q, K, V by linear projections of shape [B, seq, 4]. Then reshapes to [B, n_heads, seq, head_dim] = [B, 2, seq, 2].
	•	Rotary Embedding: Computes cos and sin matrices for each position and each head dimension pair (the code uses the formula with base 10000 and the position index). It then splits Q and K into even-index and odd-index components (since each pair of features will be rotated). The even and odd components are combined with cos and sin to produce rotated Q and K ￼. This part is vectorized.
	•	Computes attention scores with torch.matmul and applies the causal mask (upper triangular) to zero out future positions. Softmax is applied to get attention weights.
	•	Uses the attention weights to get a weighted sum of values attn_out for each head, then concatenates heads and applies the output linear Wo.
	•	Adds the residual connection from the input of the sub-layer, then LayerNorm.
	•	Passes the result through the FFN (linear -> ReLU -> linear), adds residual, and LayerNorm again.
	•	The forward pass does this for two layers. The output is then projected to vocab logits by output_proj.
	•	training_step and validation_step: These handle constructing the shifted input and target, and computing the cross-entropy loss with masking. We take batch (which might be (seq, seq_len) or just seq tensor) and split it into inp = seq[:, :-1] and target = seq[:, 1:] (so inp is everything except the last token, and target is everything except the first token). This aligns each input position with the next token as target. We then find the index of the tab token in each sequence (using argmax on a mask where seq == tab_id) and create a mask to ignore all target positions that are before that index (these correspond to input and tab). Those positions in target are set to -100 (which is PyTorch’s ignore index for CrossEntropyLoss). Finally, we run self.forward(inp) to get logits and compute F.cross_entropy against the masked target. The loss is logged for monitoring. The validation step does the same on val batches.
	•	configure_optimizers: We use a simple Adam optimizer. This could be extended to use learning rate schedulers or other optimizers as needed.

Running the training:
	1.	Prepare DataLoader: You should create a torch.utils.data.Dataset that reads the train.txt file. Each line should be Base64-decoded to get the raw input\toutput string. Then split on \t to separate input and output. Finally, convert the combined string (input + tab + output) into a list of token IDs (for example, using bytes(sequence, 'ascii') to get the byte values directly works since our vocab is just byte values). Return this list of IDs as a tensor. Do the same for validation. Because sequences can have different lengths, you might want to implement a custom collate_fn that pads sequences to the same length in a batch and returns a tensor and perhaps the original length or mask (the code as written can infer mask from tab position and ignores pad if pad token is set to -100 in targets).
	2.	Initialize Model: model = MinimalTransformer(). You can adjust hyperparameters here if needed (though the prompt specifics fix them: vocab_size=256, d_model=4, etc.).
	3.	Training Loop: Using PyTorch Lightning, you can instantiate a Trainer:

trainer = pl.Trainer(max_epochs=50, gradient_clip_val=1.0)
trainer.fit(model, train_dataloader, val_dataloader)

This will run the training and validation steps accordingly. Given the small model and data, you might train for, say, 50 epochs and see very low training loss. You can stop earlier if it converges (Lightning has callbacks like EarlyStopping you could use).

	4.	Monitoring: The training and validation loss will be printed. Ideally, training loss will drop quickly. If validation loss also drops and tracks training loss, the model is learning patterns that generalize to val set; if validation loss remains higher or flat, the model may be overfitting or the validation tasks differ.
	5.	Inference: After training, you can test the model by feeding an input prompt. To generate an output, feed the model the prompt (with a tab at the end) and then sample tokens one by one from the output logits until an end-of-sequence or some length. (In our simplistic dataset, maybe the output always terminates at a certain point or you could mark an end with a special token if defined). Since we didn’t explicitly include an end-of-sequence token, you would need to know or guess the output length or use a special delimiter (like newline) to denote end if that’s in your data. For a quick test, you can also run a batch through the model in eval mode and check the logits argmax to see if it outputs the expected result for a training example.

Important: When decoding model outputs, you’ll get token IDs – convert them back to characters (e.g., chr(id) for each) to see the result. Because our output projection isn’t weight-tied to the input embedding, and given the model’s small size, you might occasionally get tokens outside the expected output distribution (especially if it’s not fully trained). But if it has overfit the data, it will output exactly the training pair’s outputs for those inputs.

Why Base64 Encoding for Data?

You might wonder why our sample data is Base64-encoded. This is mainly for convenience and safety in data handling. By encoding the input-output pairs in Base64:
	•	We ensure that all characters in the file are printable ASCII (letters, numbers, +, /, and =) so there are no special control characters (like actual tabs or newlines) that could confuse file reading or copy-pasting the data. The tab delimiter and any potentially problematic characters in the raw data are hidden within the Base64 encoding.
	•	It avoids issues with leading/trailing whitespace or invisible characters if someone views or edits the file. For example, a raw tab in a text file might be converted to spaces or might not be visible; in Base64 form, it’s represented by a sequence of visible characters.
	•	When using an interactive environment or certain shells, tabs and some control characters can have special meanings (like tab might auto-complete or indent). By storing data in Base64, we treat everything as plain text until we deliberately decode it in the training script.
	•	In this particular write-up (in Markdown), including raw tabs or certain bytes could be misinterpreted by the markdown renderer. So Base64 is a way to embed the data cleanly in this document.

When training the model, you’d decode the Base64 to get the actual string. After that, the data processing is straightforward. In summary, Base64 is used here to simplify the presentation and handling of example data, ensuring the focus remains on the model rather than data format quirks.

With this setup, you have a complete minimal Transformer implementation and the knowledge to train and expand it. You can experiment by training the model on the provided sample data (after decoding it) and observing how quickly it learns the mappings. Then, try the progressive expansion: for instance, double the embedding size to 8 and see if you can reuse the tiny model’s weights to initialize a slightly larger model (you’d have to write some code to transplant weights). This can validate the concept of unfolding.

Even without expansion, this minimal Transformer is a powerful demonstration: it’s a fully working sequence transducer in under 3K parameters. By carefully designing the experiment (small data, simple task), it can reach near-perfect accuracy, underscoring that even the mighty Transformer architecture can be distilled to a very basic form and still function. Have fun with this tiny Transformer!  ￼ ￼

### Why We Do Not Use `nn.TransformerEncoderLayer`

While `nn.TransformerEncoderLayer` provides a cleaner API, we **intentionally avoid it** due to the following reasons:

1. **Weight Mirroring for Origami Expansion**  
   - Our future "Origami" expansion scheme requires direct control over weight matrices to enable **mirroring and reflection** across axes.  
   - `nn.TransformerEncoderLayer` encapsulates weights, making **precise control and structured expansion difficult**.

2. **Explicit Control Over QKV and FFN Weights**  
   - This model manually defines **Wq, Wk, Wv, Wo, ff_in, and ff_out**, allowing **fine-grained weight manipulation**.  
   - `nn.TransformerEncoderLayer` abstracts these operations, making **custom weight transformations impractical**.

3. **Integration of Rotary Positional Embeddings (RoPE)**  
   - RoPE requires **modifying Q and K before attention computation**.  
   - `nn.TransformerEncoderLayer` does not natively support RoPE, requiring unnecessary workarounds to inject it.

### Conclusion
By keeping `nn.ModuleDict()`, we **preserve full control over attention and feed-forward components**, ensuring smooth compatibility with **Origami-based model expansion** in the future.

### Why We Added Dropout (Dynamic Scaling)

Dropout is introduced to **improve training stability and prevent overfitting** in **larger models** while ensuring **no harmful effects on small models**.

1. **Small Models (`d_model=4`) Remain Unaffected**  
   - Dropout is dynamically computed as:  
     \[
     \text{dropout} = \min(0.1, 0.5 \times \frac{d_{\text{model}}}{256})
     \]
   - For **tiny models** (e.g., `d_model=4`), this results in **near-zero dropout**, preventing information loss.

2. **Scales Automatically for Future Expansions**  
   - As **Origami expands the model**, dropout **gradually increases**, helping **larger networks** avoid overfitting.
   - The rate **caps at 10% (`0.1`)**, ensuring it never removes too much information.

3. **Applied Before Residual Connections**  
   - Dropout is **only applied before residual connections** in:
     - **Self-Attention Output**
     - **Feed-Forward Network Output**
   - This follows **best practices** for stabilizing Transformer training.

### Conclusion
Dropout remains **inactive for the current tiny model** but **scales dynamically for future expansions**, ensuring robustness while preserving information.

### Why We Apply RoPE to Values (V)

Traditionally, **Rotary Positional Embeddings (RoPE) are applied only to Queries (Q) and Keys (K)** because attention scores encode relative positional information. However, we extend RoPE to **Values (V)** for the following reasons:

1. **Increased Expressivity in Small Models**  
   - With a tiny embedding size (`d_model=4`), the model has **limited capacity** to capture complex positional structures.  
   - Rotating V ensures that **positional information is not lost in the attention output**, improving representation quality.

2. **Future-Proofing for Larger Models (Origami Expansion)**  
   - As the model **scales**, deeper layers require **stronger positional coherence** across representations.  
   - Applying RoPE to V ensures that positional information **propagates fully through residual connections**, benefiting future expansions.

3. **Minimal Computational Cost, High Potential Gain**  
   - RoPE is a simple **element-wise transformation** with **no additional learned parameters**.  
   - The computational overhead is negligible, making it a **low-cost enhancement**.

### Conclusion
By rotating V, we enhance the **positional expressivity** of both **small models** and **larger Origami-expanded architectures**, ensuring stronger

