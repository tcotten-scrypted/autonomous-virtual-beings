# Fluctlight Transformer 4-Token Context Window Results

## Training Metrics
- Training loss: 5.5e-6 (extremely low)
- Validation accuracy: 100% (perfect)
- Validation loss: 1.14e-5 (extremely low)

## Model Configuration
- Vocabulary size: 256
- Embedding dimension (d_model): 4
- Attention heads: 2
- Number of layers: 2
- Feed-forward dimension: 8
- Context window: 4

## Test Results

| Input | Max Length | Generated | Analysis |
|-------|------------|-----------|----------|
| `a`   | 1          | `A`       | Basic capitalization works |
| `aaAa`| 1          | `A`       | Single token prediction is 'A' |
| `aaAa`| 4          | `AAAA`    | Generates four 'A's |
| `ajAa`| 4          | `AAAA`    | Generates four 'A's despite out-of-distribution 'j' |
| `ajJa`| 4          | `AAAA`    | Generates four 'A's despite out-of-distribution 'j' and 'J' |
| `k`   | 1          | `A`       | Generates 'A' for an out-of-distribution character |
| `k`   | 4          | `AAAA`    | Generates four 'A's for an out-of-distribution character |
| `ordo`| 1          | `A`       | Generates 'A' for completely out-of-distribution input |

## Analysis

The model has overfitted to a simple pattern: output "A" for every position, regardless of input. This is evident from:

1. Perfect training and validation metrics (near-zero loss)
2. Consistent "A" output regardless of input characters
3. No differentiation between in-distribution ('a', 'A') and out-of-distribution (all other characters)

The model has essentially learned the most frequent output token in the training data ("A") and decided to produce it for all inputs. This is a classic case of the model finding the simplest solution that minimizes loss on the training data.

While this behavior perfectly satisfies the capitalization task for 'a', it doesn't demonstrate true understanding of the capitalization concept. The model has memorized that the correct output is always "A" rather than learning the lowercase→uppercase transformation rule.

This suggests that:
1. The training data was too narrowly focused on a single character
2. The model found a shortcut solution rather than learning a generalizable rule
3. For more complex learning, the training data should include more character varieties

For future experiments, expanding the character set beyond just 'a/A' would force the model to learn the true capitalization relationship rather than memorizing a specific output pattern.
