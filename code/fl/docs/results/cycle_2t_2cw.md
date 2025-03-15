# Fluctlight Transformer Cyclic 2-Token Transformation (CW=2)
## Training Information
The model successfully learned the cyclic transformation task in a single epoch using v_scale=0.0:
- Training loss: 1.450
- Validation loss: 0.639
- Training files: `data/cycle_2t_2cw-train.txt`
- Validation files: `data/cycle_2t_2cw-val.txt`
- Test file: `data/cycle_2t_2cw-test.txt`

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Vocabulary size | 256 |
| Embedding dimension (d_model) | 4 |
| Attention heads | 2 |
| Number of layers | 2 |
| Feed-forward dimension | 8 |
| Context window | 2 |
| Dropout rate | 0.0078125 |
| Total parameters | 2.7K |

## Test Results for v_scale = 0.0

### Epoch 0 Results

| Match | Errors | RMSE | Input | Expected | Actual |
|-------|--------|------|-------|-----------|---------|
| ✅ | 0 | 0.000 | a | a | a |
| ✅ | 0 | 0.000 | a | aa | aa |
| ✅ | 0 | 0.000 | a | aaaa | aaaa |
| ✅ | 0 | 0.000 | a | aaaaaaaa | aaaaaaaa |
| ✅ | 0 | 0.000 | b | b | b |
| ✅ | 0 | 0.000 | b | bb | bb |
| ✅ | 0 | 0.000 | b | bbbb | bbbb |
| ✅ | 0 | 0.000 | b | bbbbbbbb | bbbbbbbb |
| ✅ | 0 | 0.000 | ab | a | a |
| ✅ | 0 | 0.000 | ab | ab | ab |
| ✅ | 0 | 0.000 | ab | abab | abab |
| ✅ | 0 | 0.000 | ab | abababab | abababab |
| ✅ | 0 | 0.000 | ba | b | b |
| ✅ | 0 | 0.000 | ba | ba | ba |
| ✅ | 0 | 0.000 | ba | baba | baba |
| ✅ | 0 | 0.000 | ba | babababa | babababa |

## Analysis
The model demonstrates excellent learning characteristics:
1. Perfect accuracy achieved in just one epoch
2. Zero errors across all test cases
3. Successfully learned both single-token and two-token patterns

Key Observations:
- Model achieves perfect pattern replication with v_scale = 0.0
- Model required one additional epoch with v_scale = 1.0
- Handles both single character repetition (a→aaaa) and alternating patterns (ab→abab)
- Very efficient learning with minimal architecture (only 2.7K parameters)
- Fast convergence with training loss of 1.450 dropping to validation loss of 0.639

Implications:
- Minimal transformer architecture (4-dimensional embeddings, 2 heads) is sufficient for simple pattern learning
- 2-token context window successfully captures both single-token and alternating-token patterns
- Low dropout rate (0.0078125) allows for stable pattern learning
- The model demonstrates that RoPE v-scaling is not necessary for this simple pattern recognition task

This experiment demonstrates that even a tiny transformer can perfectly learn and reproduce simple cyclic patterns with minimal computational resources and training time.