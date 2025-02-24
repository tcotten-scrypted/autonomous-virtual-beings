Minimal Transformer with RoPE – Implementation and Documentation

1. Minimal Transformer Implementation (PyTorch Lightning)

Below is a complete PyTorch Lightning implementation of a minimal Transformer model with Rotary Positional Embedding (RoPE). This model uses a vocabulary of 256 (extended ASCII), embedding size 4, two Transformer encoder layers, and two attention heads (each head of dimension 2). It includes token embedding, causal self-attention (with RoPE applied to Q & K), a feed-forward network (4 → 8 → 4), residual connections, and layer normalization. The LightningModule handles training and validation steps (using cross-entropy loss).

Training Data

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

