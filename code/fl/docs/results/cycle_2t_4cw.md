# 2-Token Pattern Learning with 4-Token Context Window

This document describes training a Fluctlight model to learn cyclic patterns using 2 active tokens (a,b) with a 4-token context window.

## Training Data Format

The training data consists of Base64-encoded input/output pairs, where:
- Inputs can be 1-4 tokens long
- Outputs are always 4 tokens long
- Shorter inputs are padded with zeros on the left
- The model must learn to handle padding appropriately

See `training/cycle_2t_4w.py` for the data generation code.

## Pattern Examples

### Single Token (Length 1)
Input tokens are padded with 3 zeros on the left:
```
'a' -> 'aaaa'  ([0,0,0,a] → repeat 'a')
'b' -> 'bbbb'  ([0,0,0,b] → repeat 'b')
```

### Double Token (Length 2)
Input tokens are padded with 2 zeros on the left:
```
'aa' -> 'aaaa'  ([0,0,a,a] → repeat 'a')
'ab' -> 'aaab'  ([0,0,a,b] → cycle to 'a' then 'b')
'ba' -> 'bbba'  ([0,0,b,a] → cycle to 'b' then 'a')
'bb' -> 'bbbb'  ([0,0,b,b] → repeat 'b')
```

### Triple Token (Length 3)
Input tokens are padded with 1 zero on the left:
```
'aaa' -> 'aaaa'  ([0,a,a,a] → repeat 'a')
'aab' -> 'aaab'  ([0,a,a,b] → cycle to 'a' then 'b')
'aba' -> 'aaba'  ([0,a,b,a] → pattern 'aaba')
'baa' -> 'bbaa'  ([0,b,a,a] → pattern 'bbaa')
'bba' -> 'bbba'  ([0,b,b,a] → cycle to 'b' then 'a')
```

### Full Length (Length 4)
No padding needed, patterns maintain themselves:
```
'aaaa' -> 'aaaa'  ([a,a,a,a] → maintain pattern)
'abab' -> 'abab'  ([a,b,a,b] → maintain alternation)
'abba' -> 'abba'  ([a,b,b,a] → maintain palindrome)
```

## Training Results

The model achieves perfect accuracy with both v_scale settings:

### With RoPE on Value Vectors (v_scale=1.0)
```
Epoch 5: val_loss=0.507
Test Results:
- Passed: 120/120 (100.0%)
- Failed: 0 (0.0%)
- Total Errors: 0
- Average RMSE: 0.000
```

### Without RoPE on Value Vectors (v_scale=0.0)
```
Similar perfect results achieved by epoch 5
```

## Key Insights

1. **Padding Handling**: The model must learn to handle left-padded zeros correctly while maintaining pattern recognition.

2. **Pattern Complexity**: The 4-token context window allows for more complex patterns than the 2-token window:
   - Single tokens expand to full 4-token repetitions
   - Double tokens establish initial patterns that complete to length 4
   - Triple tokens demonstrate transition handling
   - Full-length sequences maintain their patterns

3. **Position Awareness**: The model successfully learns position-dependent transformations despite varying input lengths and padding.

4. **RoPE Effectiveness**: Both v_scale settings achieve perfect accuracy, suggesting the model can learn these patterns with or without RoPE on value vectors.

## Generation Process

The generation process must carefully handle padding tokens when input length < context_window:

1. Right-align input with left padding: `[0,0,0,a]` for single token
2. Generate next token based on visible context
3. Shift window and continue generating
4. Pattern completion depends on both input tokens and their positions

This demonstrates the model's ability to:
- Handle variable-length inputs
- Manage padding tokens correctly
- Maintain consistent pattern generation
- Complete sequences appropriately based on context
```

This document captures the key aspects of the 2-token 4-context-window case while following the structure of the original 2-token 2-context-window documentation. Would you like me to explain or modify any part of it?
