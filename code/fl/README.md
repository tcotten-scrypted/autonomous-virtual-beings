<img src="assets/fluctlight-badge.svg" alt="Fluctlight Logo" width="200" height="200" align="right"/>

# Fluctlight: Minimal Transformer with RoPE

A modern Python implementation of a Transformer model with Rotary Positional Embeddings (RoPE), as a minimally viable model capable of pattern mimicry.

> The name "Fluctlight" is inspired by Sword Art Online, where it represents the digital soul or consciousness that gives artificial beings their unique personalities and capabilities. Like its namesake, this project aims to create a minimal yet complete implementation that captures the essence of neural processing.

## Overview

Fluctlight is a minimalist implementation of the Transformer architecture that incorporates Rotary Positional Embeddings (RoPE) in an experimental matter for enhanced sequence modeling. The project demonstrates how to build and train a compact but effective pattern mimicry model.

### Experimental Goals

- Create a minimally viable model capable of pattern mimicry
- Test across multiple domains, including simulation & gaming
- Test deploying in ZK circuits such as a Cairo-based decentralized network
- Experiment for use as personality cores for AVBs

### Unproven Areas of Interest
- Train larger Origami-derived models from Fluctlights
- Explore adaptive normalization scaling in expanded models
- Test RoPE interpolation on value vectors for position awareness

### Key Features
- PyTorch-based Transformer architecture
- Rotary Positional Embeddings (RoPE)
- Rich visualization for model training and text generation
- Efficient byte-level tokenization (vocab size: 256)
- Terminal-based interactive text generation UI
- Dynamic normalization scaling for model expansion

## Model Architecture

The Fluctlight model uses the following configuration:
- Parameters: 2,656 (including final normalization layer)
- Vocabulary Size: 256 (byte-level encoding)
- Hidden Dimension: 4
- Number of Heads: 2
- Number of Layers: 2
- Head Dimension: 2 (per head)
- Context Window: 2 tokens (minimum viable for pattern learning)
- Embedding: Rotary Positional Embedding (RoPE) on Q and K
- Optional: Experimental RoPE on V vectors (disabled by default)
- Normalization: Adaptive scaling (inactive at d_model=4)

See the architecture diagrams in `docs/` for detailed visualization.

## Setup and Usage

1. Install UV & Set Up Environment:
```bash
# Install UV if you haven't already
https://docs.astral.sh/uv/getting-started/installation/

# Create virtual environment and activate it
uv venv && source .venv/bin/activate  # Unix-like
# or
uv venv && .venv\Scripts\activate  # Windows

# Install dependencies with UV (faster than pip alone)
uv pip install -r requirements.txt

# For development (optional)
uv pip install -r dev-requirements.txt
```

2. Train the model:
```bash
python -m fluctlight.cli train --train-file data/sample-train.txt --val-file data/sample-val.txt --output-dir checkpoints
```

3. Generate text (low temperature for stable patterns):
```bash
python -m fluctlight.cli generate --checkpoint checkpoints/last.ckpt --input-text "ab" --temperature 0.2
```

4. Run the interactive cycling demo:
```bash
python examples/test_cycling.py
```

## Project Structure

```
fluctlight/
├── fluctlight/        # Core implementation
├── docs/             # Documentation and diagrams
├── examples/         # Usage examples
├── tests/           # Test suite
└── data/            # Training data
```

## Implementation Notes

- The model uses byte-level tokenization, allowing it to handle any text input without a separate tokenizer
- RoPE implementation provides better handling of positional information compared to absolute positional embeddings
- The small model size (4-dimensional embeddings) demonstrates core Transformer concepts while remaining computationally efficient
- Minimal context window of 2 tokens is sufficient for learning basic patterns like "ababab"
- Adaptive normalization scaling enables smooth transition to larger models
- Temperature control is crucial for stable pattern generation (0.1-0.3 recommended)

## Empirical Evidence

- Successfully learns alternating patterns with 2-token context window
- Stable training with up to 16 active tokens from the 256-token vocabulary
- Low temperatures (0.1-0.3) produce consistent pattern extrapolation
- RoPE scaling enables position-aware attention even in minimal context
- Zero-impact normalization scaling at current size (d_model=4)

## AI Usage

AI was used to generate portions of this repository. See [AI.txt](AI.txt) for details about the AI tools and their contributions to the project.

## Author

Tim Cotten <tcotten@scryptedinc.com>
Part of the AVB (Autonomous Virtual Beings) public repository

## License

MIT License