# Fluctlight Transformer – Implementation and Documentation

## A Minimally Viable Transformer w/ Attention, Utilizing RoPE

Below is a complete PyTorch Lightning implementation of a minimal Transformer model. This model uses a vocabulary of 256 (extended ASCII), embedding size 4, two Transformer encoder layers, and two attention heads (each head of dimension 2). It includes token embedding, causal self-attention, a feed-forward network (4 → 8 → 4), residual connections, and layer normalization. The LightningModule handles training and validation steps (using cross-entropy loss).

## Project Design – Model Explanation and Usage

### Purpose and Design Philosophy

This project demonstrates a Fluctlight Transformer – a tiny Transformer model with only on the order of 2,900 parameters. The purpose is to create the simplest functional Transformer to illustrate the core principles of a viable transformer with attention capable of simple mimicry tasks. By using extremely small dimensions (embedding size 4, feed-forward hidden size 8, 2 heads, 2 layers), the model is easy to train on a CPU and inspect or even overfit on a small dataset. Starting with such a minimal model (2.9K params) provides a baseline that can be progressively expanded to larger models. This minimal design is great for efficient experimentation: it's small enough to train quickly, and it serves as a starting point for future unfolding (growing the model in size while reusing the learned parameters).

Despite its size, the model includes all key components of a Transformer:
- Token embeddings for a vocabulary of 256 (covering extended ASCII characters).
- Multi-head self-attention mechanism (2 heads, each 2-dimensional) with Rotary Positional Embeddings (RoPE) to encode token positions.
- Feed-Forward Network (a two-layer MLP with hidden size 8) at each layer.
- Residual connections and Layer Normalization applied after each sub-layer (Add & Norm).
- A final output projection to map the 4-dimensional representations back to 256-dimensional token logits.

The use case for this model is primarily experimental. It can learn very simple patterns or mappings (for example, mapping an input string to an output string in the training data). With a very small capacity, it will typically memorize the training data if trained for long, which is useful for verifying that the model and training process are working. In practice, one would not use such a tiny Transformer for real tasks, but it's a stepping stone toward gradually building larger models by unfolding this minimal model.

### Model Architecture

Key architecture hyperparameters and components:
- Vocabulary: 256 tokens, representing extended ASCII. Each character (byte) is a token. This includes standard printable ASCII and control characters.
- Embedding Dimension: 4. Each token is represented by a 4-dimensional vector. The embedding matrix thus has shape 256×4.
- Transformer Encoder Layers: 2 layers in stack. Each layer has:
  - Multi-Head Self-Attention (MHSA): 2 heads. The model dimension 4 is split into 2 heads of dimension 2 each. Each head attends to the sequence with learned query, key, value projections (Wq, Wk, Wv each of shape 4×4). Causal masking is used so that tokens only attend to previous tokens (this allows the model to be used for autoregressive generation).
  - Rotary Positional Embedding (RoPE): Applied to the Q and K vectors in attention. RoPE encodes positions by rotating Q and K in a 2D subspace for each pair of dimensions, which injects position information as a phase change in dot-product attention. This effectively means the attention scores depend on the relative positions of tokens, improving the model's ability to generalize to longer sequences without fixed positional embeddings.
  - Feed-Forward Network (FFN): A two-layer MLP applied to each token position. It expands the 4-dim representation to 8-dim (ff_in layer 4→8 with ReLU), then back down to 4-dim (ff_out layer 8→4). This gives the network capacity to transform and mix information across dimensions in a non-linear way.
  - Residual Connections: The output of the attention sub-layer is added back to its input (skip connection), and same for the FFN sub-layer.
  - Layer Normalization: After each addition (residual), a LayerNorm normalizes the features. This helps stabilize training. (We use a Post-Norm configuration: normalization is applied after each sub-layer's residual addition, as in the original Transformer.)
- Output Projection: A linear layer (4→256) maps the final 4-dimensional token representations to logits over the 256-token vocabulary. This is analogous to the inverse of the embedding layer, translating the model's learned representation back into actual character predictions. (Weights are not tied in this implementation, but they could be in theory.)

Despite the limited size, the model has all the pieces needed for sequence learning. In total, it has on the order of ~2.9K parameters (embedding tables, linear weights, layer norm parameters, etc.). For example, the token embedding matrix is 256×4 = 1024 parameters, and the output projection is another 1024; the two transformer layers together contribute roughly a few hundred parameters each (Q/K/V/O weights, FFN weights, and biases, plus LayerNorm gains/biases).

### Architecture Diagram

Below is a Mermaid diagram illustrating the model architecture and data flow through one layer of the Transformer (the model has two such layers back-to-back). This diagram shows the token embedding, the multi-head attention with RoPE applied to Q and K, the feed-forward network, residual connections, and layer normalization, as well as the final projection to output logits.

```mermaid
flowchart TD
    subgraph Input_Sequence["Input Tokens (bytes)"]
    end
    Input_Sequence --> ContextLimit["Context Window Limit (64 tokens)"]
    ContextLimit --> Embedding["Token Embedding (256 -> 4)"]
    Embedding --> Layer1["Transformer Encoder Layer 1"]
    
    subgraph Layer1
        direction LR
        subgraph MHA1["Multi-Head Attention (2-head, RoPE)"]
            direction TB
            LN1a["LayerNorm"] --> Q1["Wq"]
            LN1a --> K1["Wk"]
            LN1a --> V1["Wv"]
            Q1 --> Q1_vec["Q vectors"]
            K1 --> K1_vec["K vectors"]
            V1 --> V1_vec["V vectors"]
            Q1_vec -. "apply RoPE" .-> Q1_rope["Q (rotated)"]
            K1_vec -. "apply RoPE" .-> K1_rope["K (rotated)"]
            V1_vec -. "apply RoPE" .-> V1_rope["V (rotated)"]
            Q1_rope --> AttnScores1["Dot-Product & Softmax"]
            K1_rope --> AttnScores1
            AttnScores1 --> AttnWeights["Attention Weights"]
            AttnWeights --> AttnOut1["Weighted Sum"]
            V1_rope --> AttnOut1
            AttnOut1 --> OutProj1["Wo"]
        end
        
        OutProj1 --> Dropout1["Dropout"]
        Dropout1 --> AddRes1["Add (residual)"]
        AddRes1 --> LN2["LayerNorm"]
        LN2 --> FFN1["Feed-Forward (4 -> 8 -> 4 with ReLU)"]
        FFN1 --> Dropout2["Dropout"]
        Dropout2 --> AddRes2["Add (residual)"]
        AddRes2 --> LN3["LayerNorm"]
    end
    
    Layer1 --> Layer2["Transformer Encoder Layer 2 (same structure)"]
    Layer2 --> OutputProj["Output Projection (4 -> 256)"]
    OutputProj --> Logits["Logits (prediction scores)"]
```

Diagram Explanation: Each input token (a byte character) is first mapped to a 4-dimensional embedding. In Multi-Head Attention, the model computes Query, Key, and Value vectors (each 4-dim, split across 2 heads) for each position. Rotary Positional Encoding (RoPE) is applied to Q and K vectors, rotating them in a plane by an angle proportional to their position index. The dot-product attention then produces weighted sums of values from previous positions for each token (since causal masking prevents looking ahead). The attention output goes through an output linear Wo and is added back to the input (residual), followed by a LayerNorm. Next, the Feed-Forward Network processes each token's data independently through a ReLU-expanded 8-dim hidden layer and back to 4-dim, followed by another residual add and LayerNorm. After two such layers, the final normalized 4-dim vectors are projected to 256-dimensional logits, one for each possible token, determining the predicted output characters.

## Fluctlight Code Architecture

```mermaid
graph TD
 subgraph CLI
 cli[cli.py] --> train[Train Command]
 cli --> generate[Generate Command]
 end
 subgraph Core
 model[model.py<br/>FluctlightTransformer] --> dataset[dataset.py<br/>Base64Dataset]
 model --> utils[utils.py<br/>Base64 Utils]
 end
 subgraph Testing
 test_model[test_model.py] --> model
 test_dataset[test_dataset.py] --> dataset
 test_utils[test_utils.py] --> utils
 test_device[test_device.py] --> model
 end
 train --> model
 generate --> model
 dataset --> utils
```

### Overview
The Fluctlight project implements a byte-level transformer model with Rotary Position Embeddings (RoPE). The architecture focuses on efficiency and clarity while maintaining core transformer functionality.

### Core Components
#### FluctlightTransformer
The main model implementation with the following architecture:
- Vocabulary: 256 tokens (byte-level encoding)
- Embedding Dimension: 4 (compact but effective)
- Attention Heads: 2 (each head dimension: 2)
- Feed-forward Dimension: 8 (2x embedding dimension)
- Context Window: 64 tokens
- Position Encoding: Rotary Positional Embedding (RoPE)

Key features:
- Byte-level tokenization eliminates need for complex tokenizer
- RoPE for enhanced position-aware attention
- Dynamic dropout based on model size
- Efficient context window management

#### Dataset Handling
The `Base64Dataset` class provides:
- Loading of base64-encoded input-output pairs
- Optional prepending of training data
- Automatic device placement
- Efficient sequence collation and padding

Data format:
```
base64(input)\tbase64(output)\n
```

#### Training Infrastructure
Components for efficient training:
- Automatic device detection (CUDA, MPS, CPU)
- Configurable CPU worker allocation
- Batch collation with padding
- Context window enforcement

### Implementation Details
#### Attention Mechanism
The attention implementation uses:
1. RoPE for positional information
2. Causal masking for autoregressive prediction
3. Multi-head attention with efficient head dimension splitting

#### Training Process
The training loop:
1. Loads base64-encoded pairs
2. Applies context window limits
3. Shifts sequences for next-token prediction
4. Computes loss with proper padding handling

#### Utility Functions
Core utilities:
- Base64 decoding for training data
- Device detection and management
- DataLoader creation with optimal settings
- Sequence collation and padding

### Testing
The test suite covers:
1. Model architecture and forward pass
2. Dataset loading and processing
3. Device handling and tensor placement
4. Training functionality
5. Utility functions

### File Structure
```
fluctlight/
├── model.py # FluctlightTransformer implementation
├── dataset.py # Data loading and processing
├── utils.py # Utility functions
└── cli.py # Command-line interface
tests/
├── test_model.py # Model tests
├── test_dataset.py # Dataset tests
├── test_device.py # Device handling tests
└── test_utils.py # Utility function tests
```

### Progressive Expansion (Future Unfolding of the Model)

One interesting aspect of this minimalist model is that it is the basis for experiments in progressively "unfolding" or expanding weights to a larger Transformer by mirroring/copying its weights. The idea is to use the small model as a building block and increase capacity without starting from scratch. This approach is inspired by function-preserving transformations like Net2Net which introduced methods to expand neural networks (width or depth) while initializing the larger model to behave exactly like the smaller one.

Our goal is to train the minimal model on a simple task, then gradually grow it (e.g., double the embedding to 8, 16, … add more heads and layers) to handle more complex tasks, each time reusing the previous weights as a starting point. This approach treats the small model as a "seed" that can blossom into a larger model, a concept sometimes referred to as model growth or model folding. The minimal model's simplicity and low parameter count (2.9K) make it feasible to experiment with such growth quickly.

Why start so small (2,904 parameters)? Starting with a minimal number of parameters ensures that the model can memorize small datasets easily and that every parameter's role can be inspected. It reduces training time to seconds and allows observing training dynamics on a micro-scale. It also forces us to include only the most essential components of a Transformer. From this base, every time we increase capacity, we understand exactly what new parameters are added. This stepwise expansion helps in demystifying how each part of a Transformer contributes to its performance. In essence, the minimal model is like a bonsai tree – small but fully formed – which can be replanted into a bigger pot to grow into a larger tree given time.

How RoPE helps scaling: Rotary Positional Embedding is particularly handy when scaling up sequence length or model size because it encodes positions implicitly and continuously. Unlike fixed positional embeddings (which might be learned for a specific maximum length), RoPE uses a deterministic formula to rotate Q/K vectors. This means if we increase the sequence length, we don't need new position embeddings – the same formula extrapolates to unseen positions (to an extent). When expanding model dimensions, we can also integrate RoPE into the new dimensions without breaking the existing ones: e.g., if we double the embedding, we can assign the original RoPE frequencies to one half of the dimensions and perhaps initialize the other half with either repeated frequencies or new ones (for capturing finer positional details). The relative positioning property of RoPE means the model's attention focuses on relative distance between tokens, which tends to generalize better when the context window grows. In summary, RoPE improves the scalability of the model in terms of sequence length and can be smoothly adopted as we increase model size, without having to re-learn positional encodings from scratch for the new model.

### Why We Do Not Use `nn.TransformerEncoderLayer`

While `nn.TransformerEncoderLayer` provides a cleaner API, we **intentionally avoid it** due to the following reasons:

1. **Weight Mirroring for Origami Expansion**  
   - Our future "Origami" expansion scheme requires direct control over weight matrices to enable **mirroring and reflection** across axes.  
   - `nn.TransformerEncoderLayer` encapsulates weights, making **precise control and structured expansion difficult**.

2. **Explicit Control Over QKV and FFN Weights**  
   - This model manually defines **Wq, Wk, Wv, Wo, ff_in, and ff_out**, allowing **fine-grained weight manipulation**.  
   - `nn.TransformerEncoderLayer` abstracts these operations, making **custom weight transformations impractical**.

3. **Integration of Rotary Positional Embeddings (RoPE)**  
   - RoPE requires **modifying Q and K (and optionally V)before attention computation**.  
   - `nn.TransformerEncoderLayer` does not natively support RoPE, requiring unnecessary workarounds to inject it.

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