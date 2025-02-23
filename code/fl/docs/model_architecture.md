# Model Architecture

```mermaid
graph TD
    subgraph Input
        input[Input Tokens] --> embed[Token Embeddings<br/>dim=4]
    end

    subgraph TransformerLayers
        rope[RoPE<br/>Positional Encoding] --> attn[Self Attention<br/>heads=2<br/>head_dim=2]
        attn --> ffn[Feed Forward<br/>dim=4]
    end

    subgraph Output
        ffn --> out[Output Logits<br/>vocab=256]
        out --> sample[Sampling<br/>temperature=0.8]
    end

    embed --> rope
    ffn --> rope
    
    style input fill:#f9f,stroke:#333,stroke-width:2px
    style out fill:#9ff,stroke:#333,stroke-width:2px
    style rope fill:#ff9,stroke:#333,stroke-width:2px
```

## Architectural Details

1. **Input Processing**
   - Byte-level tokenization (vocab size = 256)
   - 4-dimensional token embeddings

2. **Rotary Positional Embeddings (RoPE)**
   - Applied to queries and keys
   - Provides relative positional information
   - Maximum sequence length: 64 tokens

3. **Transformer Layer**
   - Multi-head attention (2 heads)
   - Head dimension: 2
   - Feed-forward dimension: 4
   - Layer normalization

4. **Output Generation**
   - Logits over 256 vocabulary items
   - Temperature-based sampling (default: 0.8)

The model is intentionally compact while maintaining the essential Transformer architecture components. This design allows for efficient training and inference while demonstrating the key concepts of attention mechanisms and positional embeddings.

### Key Features

- **RoPE Benefits**: Better handling of relative positions compared to absolute positional embeddings
- **Small But Complete**: Contains all essential Transformer components in a minimal form
- **Efficient Processing**: Small dimensions allow quick training and inference
- **Byte-Level Operations**: Can handle any text input without special tokenization
