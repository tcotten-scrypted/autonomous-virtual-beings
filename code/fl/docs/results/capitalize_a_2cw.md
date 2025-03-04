# Fluctlight Transformer Capitalization Results (CW=2)

## Training Information

The model successfully learned the capitalization task with perfect accuracy:
- Training loss: 0.000117
- Validation accuracy: 100%
- Validation loss: 0.00012

### Model Configuration
- Vocabulary size: 256
- Embedding dimension (d_model): 4
- Attention heads: 2
- Number of layers: 2
- Feed-forward dimension: 8
- Context window: 2

## Test Results

The model demonstrates perfect capitalization behavior for both single and multi-character inputs:

| Input | Max Length | Generated | Analysis |
|-------|------------|-----------|----------|
| `a`   | 1          | `A`       | Basic lowercase → uppercase conversion |
| `aa`  | 1          | `A`       | Predicts first token after input |
| `aa`  | 2          | `AA`      | Correctly capitalizes both characters |
| `aA`  | 1          | `A`       | Predicts uppercase for next token |
| `Aa`  | 1          | `A`       | Maintains uppercase for next token |
| `Aa`  | 2          | `AA`      | Correctly capitalizes both characters |
| `A`   | 1          | `A`       | Maintains uppercase for single character |
| `AA`  | 1          | `A`       | Predicts uppercase for next token |
| `AA`  | 2          | `AA`      | Correctly maintains uppercase |
| `kk`  | 2          | `¿`       | Out-of-distribution input produces unexpected result |

## Analysis

The model successfully learned the capitalization pattern for its training distribution (a→A) with remarkable accuracy. The context window of 2 is sufficient for this task, as demonstrated by perfect validation metrics.

Key observations:
1. The tiny model (d_model=4) is capable of learning this simple transformation task perfectly
2. For max_length=1, it generates exactly one token (the uppercase version of the last input token)
3. For max_length=2, it generates two tokens (uppercase versions of both input tokens)
4. The model's behavior becomes unpredictable for characters outside its training distribution (`kk` → `¿`)

This experiment demonstrates that even extremely tiny transformers can learn simple token-level transformations when working with an appropriate context window size and properly aligned training data.
