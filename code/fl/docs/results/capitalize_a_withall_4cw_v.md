# Fluctlight Transformer V-Rotation Results (CW=4)

## Training Metrics
- Training loss: 0.00416
- Validation accuracy: 97.6%
- Validation loss: 0.175

## Test Results

| Input | Max Length | Generated | Analysis |
|-------|------------|-----------|----------|
| `orAo`| 4          | `oooo`    | Repeats last character |
| `orAa`| 4          | `AAAA`    | Capitalizes 'a' and repeats |
| `o5Aa`| 4          | `AAAA`    | Capitalizes 'a' and repeats |
| `o5A5`| 4          | `5555`    | Repeats last character |
| `a`   | 4          | `AAAA`    | Capitalizes 'a' and repeats |
| `AA`  | 4          | `AAAA`    | Preserves 'A' and repeats |
| `0`   | 4          | `0000`    | Preserves '0' and repeats |

## Analysis

With V-rotation enabled, the model still exhibits very similar behavior to before:

1. **Last-character fixation**: The model continues to focus primarily on the last character of the input sequence.

2. **Special 'a' → 'A' transformation**: The model correctly applies the capitalization rule when 'a' is the last character.

3. **Consistent repetition**: The model consistently repeats the predicted character for the specified max length.

The V-rotation appears to have made minimal difference in the model's behavior, with no obvious change in how it processes position information. This suggests:

1. The 'a' → 'A' transformation rule is simple enough that additional positional encoding in the value vectors isn't necessary to learn it.

2. The model's tiny size (d_model=4) may be limiting how much it can leverage the additional positional information.

3. The model's strong bias toward the last position is a simple but effective strategy that works well for this task, even with different RoPE configurations.

An interesting observation is that turning on V-rotation slightly increased the training loss (from 0.00105 to 0.00416) but slightly improved validation accuracy (from 97.0% to 97.6%). This suggests that V-rotation might be providing some regularization benefit, making the model slightly more generalizable despite being slightly harder to train.
