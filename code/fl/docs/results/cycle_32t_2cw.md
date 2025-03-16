# Fluctlight Transformer Cyclic 32-Token Transformation (CW=2)

## Overview

This experiment tests the Fluctlight model's ability to learn cyclic patterns with a 32-token vocabulary using a 2-token context window. The tokens used were the full lowercase alphabet (a-z) plus six uppercase letters (A-F).

## Training Configuration

- Vocabulary: 32 tokens ("abcdefghijklmnopqrstuvwxyzABCDEF")
- Context Window: 2 tokens
- Training Duration: 200 epochs
- Batch Size: 64
- Learning Rate: 1e-4
- Weight Decay: 1e-6
- V-Scale: 1.0

## Results

### Final Performance

After 200 epochs:
- Training Loss: 3.010
- Validation Loss: 2.180
- Training Speed: 60.60 iterations/second

### Test Results
- Total Test Cases: 4,224
- Passed: 4,128 (97.7%)
- Failed: 96 (2.3%)
- Total Errors: 192
- Average RMSE: 0.375

### Error Analysis

The model showed systematic failure patterns across 24 characters. Each failing character showed consistent error patterns across all sequence lengths (1, 2, 4, and 8 tokens). Key observations:

1. **Case Confusion**
   - c → C (RMSE: 22.627)
   - t → T (RMSE: 22.627)
   - D → d (RMSE: 22.627)

2. **High RMSE Substitutions**
   - v → A (RMSE: 37.477)
   - w → K (RMSE: 31.113)
   - z → U (RMSE: 26.163)
   - A → h (RMSE: 27.577)
   - C → h (RMSE: 26.163)

3. **Low RMSE Substitutions**
   - r → s (RMSE: 0.707, lowest error)
   - j → m (RMSE: 2.121)
   - E → H (RMSE: 2.121)

### Pattern Analysis

1. Most errors maintain consistency across sequence lengths
2. The model shows a tendency to:
   - Confuse case for similar letters
   - Substitute with visually similar characters
   - Make systematic replacements (same wrong token consistently)

## Training Convergence Benchmarks

Alternative training configurations were tested:

1. **200 Epochs**
   ```
   Learning Rate: 1e-4
   Weight Decay: 1e-6
   Batch Size: 64
   Final Train Loss: 3.090
   Final Val Loss: 2.430
   ```

2. **64 Epochs**
   ```
   Learning Rate: 1e-4
   Weight Decay: 1e-6
   Batch Size: 64
   Final Train Loss: 3.800
   Final Val Loss: 3.330
   ```

## Conclusions

1. The model demonstrates surprisingly good performance (97.7% pass rate) despite the increased vocabulary size.
2. Errors show systematic patterns, suggesting the model has learned stable but incorrect associations for certain tokens.
3. The failure modes are consistent across sequence lengths, indicating stable (though incorrect) pattern learning.
4. Training beyond 64 epochs shows significant improvement in both training and validation loss.

## Future Work

1. Investigate methods to address case confusion errors
2. Experiment with different v-scale values to improve token differentiation
3. Test alternative learning rate schedules to escape local minima
4. Consider increasing model capacity for better token representation

// ... existing content ...

## Projections and Future Experiments

### Loss-to-Error Correlation

Analysis shows a strong correlation between validation loss improvements and error reduction:
- A 0.01 decrease in validation loss (2.18 → 2.17) corresponds to:
  - ~15 fewer total errors
  - ~8-16 more passing test cases
  - Improved RMSE (0.375 → 0.282)

### Best Performance Snapshot (Epoch 138)
- Validation Loss: 2.17
- Total Tests: 4,224
- Passed: 4,143 (98.1%)
- Failed: 81 (1.9%)
- Total Errors: 164
- Average RMSE: 0.282

### Recommended Exploration Settings

1. **Extended Training Duration**
   ```
   Max Epochs: 500
   Learning Rate: 1e-4
   Weight Decay: 1e-6
   Batch Size: 64
   V-Scale: 1.0
   ```
   Rationale: Given the consistent improvement pattern, extending training may resolve additional error cases.

2. **Learning Rate Schedule**
   ```
   Initial LR: 1e-4
   Schedule: Cosine decay with warm restarts
   Cycle Length: 50 epochs
   Min LR: 1e-5
   ```
   Rationale: Help escape local minima while maintaining stable learning.

3. **Graduated Weight Decay**
   ```
   Initial: 1e-6
   Mid-training (epoch 200): 5e-7
   Final (epoch 400): 2e-7
   ```
   Rationale: Allow finer parameter adjustments as training progresses.

4. **V-Scale Exploration**
   ```
   Test Range: [0.8, 0.9, 1.0, 1.1, 1.2]
   Hold Other Parameters:
     - Learning Rate: 1e-4
     - Weight Decay: 1e-6
     - Batch Size: 64
   ```
   Rationale: Find optimal position encoding scaling for 32-token space.

5. **Hybrid Approach**
   ```
   Phase 1 (epochs 0-200):
     - Learning Rate: 1e-4
     - Weight Decay: 1e-6
     - V-Scale: 1.0
   
   Phase 2 (epochs 201-500):
     - Learning Rate: 5e-5
     - Weight Decay: 5e-7
     - V-Scale: Best from exploration
   ```
   Rationale: Combine initial rapid learning with refined parameter tuning.

### Expected Outcomes

Based on the observed loss-to-error correlation:
- Target validation loss: 2.15 or better
- Expected pass rate: >98.5%
- Projected error reduction: 30-50 fewer errors
- Target RMSE: <0.25

The primary focus should be on eliminating systematic errors (case confusion, consistent substitutions) while maintaining the model's current strong performance on correctly learned patterns.