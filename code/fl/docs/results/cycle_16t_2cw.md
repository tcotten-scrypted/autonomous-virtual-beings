# Fluctlight Transformer Cyclic 16-Token Transformation (CW=2)

## Experimental Setup
- Model: Fluctlight Transformer (2.7K parameters)
- Context Window: 2 tokens
- Training Method: Includes vocabulary seeding
- Test Cases: 1088 total patterns
  - 64 single token patterns (16 tokens × 4 lengths)
  - 1024 two token patterns (16² combinations × 4 lengths)

## Training Configurations Tested

| Configuration | Weight Decay | V-Scale | Train Loss | Val Loss | Outcome |
|--------------|--------------|----------|------------|-----------|---------|
| Default | 1e-5 | 1.0 | 2.170 | 1.520 | Partial Success |
| Reduced WD | 1e-6 | 1.0 | 2.180 | 1.450 | PASSES |
| No RoPE | 1e-6 | 0.0 | 2.520 | 1.760 | FAILS (210 errors) |

## Analysis of Results

### 1. Weight Decay Impact (v-scale=1.0)
- **1e-5 (Default)**:
  - Partial convergence
  - Better training loss (2.170)
  - Higher validation loss (1.520)
  - Shows systematic failures in specific tokens

- **1e-6 (Reduced)**:
  - Full convergence
  - Slightly worse training loss (2.180)
  - Better validation loss (1.450)
  - Achieves perfect pattern replication

### 2. V-Scale Impact (weight-decay=1e-6)
- **v-scale=1.0**:
  - Successful convergence
  - Lower losses overall
  - Perfect pattern replication
  - Stable training dynamics

- **v-scale=0.0**:
  - Failed to converge
  - Significantly higher losses
  - 210/1088 test errors (19.3% error rate)
  - Shows systematic pattern breakdown

### 3. Pattern Analysis (Default Configuration)
Single Token Performance:
✅ Perfect: a,b,e,f,h,l,n (7/16)
❌ Failed: c,d,g,i,j,k,m,o,p (9/16)


Error Characteristics:
- Token substitution (e.g., o→b, p→c)
- Pattern maintenance despite wrong token
- Higher RMSE for distant token substitutions
- Consistent error patterns across lengths

## Key Findings

1. **Parameter Sensitivity**
   - Weight decay critical for 16-token learning
   - 1e-6 provides better generalization than 1e-5
   - RoPE scaling essential for convergence

2. **Training Dynamics**
   - Vocabulary seeding improves training speed
   - Pattern learning hierarchical (some tokens learn first)
   - Error patterns show structured misconvergence

3. **Scaling Characteristics**
   - 16-token space significantly more challenging than 8
   - RoPE scaling becomes more critical with token count
   - Weight decay needs adjustment for larger token spaces

## Conclusions

1. The optimal configuration for 16-token learning is:
   - Weight decay: 1e-6
   - V-scale: 1.0
   - Vocabulary seeding: Enabled

2. The model demonstrates:
   - Sensitivity to hyperparameters increases with token count
   - Clear phase transition with proper parameter settings
   - Structured failure modes when suboptimal

3. Future considerations:
   - Investigate intermediate v-scale values
   - Explore adaptive weight decay schedules
   - Consider token embedding distance metrics

This experiment reveals the delicate balance between regularization and representation capacity needed for larger token spaces, with RoPE scaling playing a crucial role in successful convergence.
