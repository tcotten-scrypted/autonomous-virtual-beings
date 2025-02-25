# Model Architecture

```mermaid
graph TD
 subgraph Input
 input[Input Tokens] --> embed[Token Embeddings<br/>dim=4]
 end
 subgraph TransformerLayers
 layer1[Layer 1] --> layer2[Layer 2]
 
 subgraph layer1[Layer 1]
   ln1_1[LayerNorm] --> qkv1[Q/K/V Projections]
   qkv1 --> rope1[RoPE Application]
   rope1 --> attn1[Self Attention<br/>heads=2<br/>head_dim=2]
   attn1 --> wo1[Output Projection]
   wo1 --> drop1[Dropout]
   drop1 --> add1[Residual Connection]
   add1 --> ln2_1[LayerNorm]
   ln2_1 --> ffn1[Feed Forward<br/>dim=4→8→4]
   ffn1 --> drop2[Dropout] 
   drop2 --> add2[Residual Connection]
 end
 
 subgraph layer2[Layer 2]
   ln1_2[LayerNorm] --> qkv2[Q/K/V Projections]
   qkv2 --> rope2[RoPE Application]
   rope2 --> attn2[Self Attention<br/>heads=2<br/>head_dim=2]
   attn2 --> wo2[Output Projection]
   wo2 --> drop3[Dropout]
   drop3 --> add3[Residual Connection]
   add3 --> ln2_2[LayerNorm]
   ln2_2 --> ffn2[Feed Forward<br/>dim=4→8→4]
   ffn2 --> drop4[Dropout]
   drop4 --> add4[Residual Connection]
 end
 end
 subgraph Output
 add4 --> proj[Output Projection<br/>dim=4→256]
 proj --> logits[Logits<br/>vocab=256]
 end
 embed --> ln1_1
 layer2 --> proj
 
 style input fill:#f9f,stroke:#333,stroke-width:2px
 style logits fill:#9ff,stroke:#333,stroke-width:2px
 style rope1 fill:#ff9,stroke:#333,stroke-width:2px
 style rope2 fill:#ff9,stroke:#333,stroke-width:2px
```

## Architectural Details

1. **Input Processing**
   - Byte-level tokenization (vocab size = 256)
   - 4-dimensional token embeddings
   - Context window limited to 64 tokens

2. **Transformer Layers (2x)**
   - **Layer Normalization**
     - Applied before attention and feed-forward operations
   - **Multi-head Self-Attention**
     - 2 heads with head dimension = 2
     - Q, K, V projections (4→4 dimensions)
     - Output projection (4→4 dimensions)
   - **Rotary Positional Embeddings (RoPE)**
     - Applied to queries, keys, and values
     - Provides relative positional information
   - **Causal Masking**
     - Ensures tokens only attend to previous positions
   - **Feed-Forward Network**
     - Two-layer MLP (4→8→4)
     - ReLU activation between layers
   - **Residual Connections**
     - Applied after attention and feed-forward blocks
   - **Dropout**
     - Dynamic rate based on model size: min(0.1, 0.5 * (d_model / 256))
     - Applied before residual connections

3. **Output Generation**
   - Linear projection from 4→256 dimensions
   - Logits over entire byte vocabulary

## Key Features

- **Minimal Yet Complete**: Contains all essential Transformer components in just ~2.9K parameters
- **Dynamic Dropout Scaling**: Automatically adjusts dropout rate based on model size, supporting future expansions
- **RoPE Benefits**: Enhanced handling of relative positions with rotary embeddings
- **Byte-Level Operations**: Works with any text input without special tokenization
- **Device Optimization**: Automatically detects and uses the optimal available device (MPS, CUDA, or CPU)
- **Context Management**: Enforces a 64-token context window by truncating longer sequences

The model is intentionally compact while maintaining the essential Transformer architecture components. This design allows for efficient experimentation and serves as a foundation for progressive model expansion through weight mirroring techniques.