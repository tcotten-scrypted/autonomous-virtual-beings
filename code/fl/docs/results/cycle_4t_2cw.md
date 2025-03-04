Epoch 10:

| Input | Expected  | v_scale=0.0 | Correct (0.0) | v_scale=1.0 | Correct (1.0) |
|-------|----------|------------|--------------|------------|--------------|
| a     | aaaaaaaa | aaaaaaaa   | ✅          | aaaaaaaa   | ✅          |
| b     | bbbbbbbb | bbbbbbbb   | ✅          | bbbbbbbb   | ✅          |
| c     | cccccccc | cccccccc   | ✅          | cccccccc   | ✅          |
| d     | dddddddd | dddddddd   | ✅          | dddddddd   | ✅          |
| aa    | aaaaaaaa | aaaaaaaa   | ✅          | aaaaaaaa   | ✅          |
| ab    | abababab | abababab   | ✅          | abababab   | ✅          |
| ac    | acacacac | acacacac   | ✅          | bcbcbcbc   | ❌          |
| ad    | adadadad | adadadad   | ✅          | adadadad   | ✅          |
| ba    | babababa | babababa   | ✅          | babababa   | ✅          |
| bb    | bbbbbbbb | bbbbbbbb   | ✅          | bbbbbbbb   | ✅          |
| bc    | bcbcbcbc | bcbcbcbc   | ✅          | bcbcbcbc   | ✅          |
| bd    | bdbdbdbd | cdcdcdcd   | ❌          | bdbdbdbd   | ✅          |
| ca    | cacacaca | cacacaca   | ✅          | cbcbcbcb   | ❌          |
| cb    | cbcbcbcb | cbcbcbcb   | ✅          | cbcbcbcb   | ✅          |
| cc    | cccccccc | cccccccc   | ✅          | cccccccc   | ✅          |
| cd    | cdcdcdcd | cdcdcdcd   | ✅          | cdcdcdcd   | ✅          |
| da    | dadadada | dadadada   | ✅          | dadadada   | ✅          |
| db    | dbdbdbdb | dcdcdcdc   | ❌          | dbdbdbdb   | ✅          |
| dc    | dcdcdcdc | dcdcdcdc   | ✅          | dcdcdcdc   | ✅          |
| dd    | dddddddd | dddddddd   | ✅          | dddddddd   | ✅          |


Applying RoPE on V with 1.0 doesn't lower errors, but just rotates the errors.

Strangely, epoch 20 is worse than epoch 10.

| Input | Expected  | v_scale=0.0 | Correct (0.0) | v_scale=1.0 | Correct (1.0) |
|-------|----------|------------|--------------|------------|--------------|
| a     | aaaaaaaa | aaaaaaaa   | ✅          | aaaaaaaa   | ✅          |
| b     | bbbbbbbb | bbbbbbbb   | ✅          | bbbbbbbb   | ✅          |
| c     | cccccccc | cccccccc   | ✅          | cccccccc   | ✅          |
| d     | dddddddd | dddddddd   | ✅          | dddddddd   | ✅          |
| aa    | aaaaaaaa | aaaaaaaa   | ✅          | aaaaaaaa   | ✅          |
| ab    | abababab | abababab   | ✅          | abababab   | ✅          |
| ac    | acacacac | bcbcbcbc   | ❌          | aaaaaaaa   | ❌          |
| ad    | adadadad | adadadad   | ✅          | adadadad   | ✅          |
| ba    | babababa | babababa   | ✅          | babababa   | ✅          |
| bb    | bbbbbbbb | bbbbbbbb   | ✅          | bbbbbbbb   | ✅          |
| bc    | bcbcbcbc | bcbcbcbc   | ✅          | bcbcbcbc   | ✅          |
| bd    | bdbdbdbd | babababa   | ❌          | dddddddd   | ❌          |
| ca    | cacacaca | cacacaca   | ❌          | aaaaaaaa   | ❌          |
| cb    | cbcbcbcb | cbcbcbcb   | ✅          | cbcbcbcb   | ✅          |
| cc    | cccccccc | cccccccc   | ✅          | cccccccc   | ✅          |
| cd    | cdcdcdcd | cdcdcdcd   | ✅          | cdcdcdcd   | ✅          |
| da    | dadadada | dadadada   | ✅          | dadadada   | ✅          |
| db    | dbdbdbdb | dcdcdcdc   | ❌          | abababab   | ❌          |
| dc    | dcdcdcdc | dcdcdcdc   | ✅          | dcdcdcdc   | ✅          |
| dd    | dddddddd | dddddddd   | ✅          | dddddddd   | ✅          |

Here’s your updated markdown table, incorporating the new error positions from **epoch 20**:

```markdown
| Input | Expected  | v_scale=0.0 | Correct (0.0) | v_scale=1.0 | Correct (1.0) |
|-------|----------|------------|--------------|------------|--------------|
| a     | aaaaaaaa | aaaaaaaa   | ✅          | aaaaaaaa   | ✅          |
| b     | bbbbbbbb | bbbbbbbb   | ✅          | bbbbbbbb   | ✅          |
| c     | cccccccc | cccccccc   | ✅          | cccccccc   | ✅          |
| d     | dddddddd | dddddddd   | ✅          | dddddddd   | ✅          |
| aa    | aaaaaaaa | aaaaaaaa   | ✅          | aaaaaaaa   | ✅          |
| ab    | abababab | abababab   | ✅          | abababab   | ✅          |
| ac    | acacacac | bcbcbcbc   | ❌          | aaaaaaaa   | ❌          |
| ad    | adadadad | adadadad   | ✅          | adadadad   | ✅          |
| ba    | babababa | babababa   | ✅          | babababa   | ✅          |
| bb    | bbbbbbbb | bbbbbbbb   | ✅          | bbbbbbbb   | ✅          |
| bc    | bcbcbcbc | bcbcbcbc   | ✅          | bcbcbcbc   | ✅          |
| bd    | bdbdbdbd | babababa   | ❌          | dddddddd   | ❌          |
| ca    | cacacaca | cacacaca   | ❌          | aaaaaaaa   | ❌          |
| cb    | cbcbcbcb | cbcbcbcb   | ✅          | cbcbcbcb   | ✅          |
| cc    | cccccccc | cccccccc   | ✅          | cccccccc   | ✅          |
| cd    | cdcdcdcd | cdcdcdcd   | ✅          | cdcdcdcd   | ✅          |
| da    | dadadada | dadadada   | ✅          | dadadada   | ✅          |
| db    | dbdbdbdb | dcdcdcdc   | ❌          | abababab   | ❌          |
| dc    | dcdcdcdc | dcdcdcdc   | ✅          | dcdcdcdc   | ✅          |
| dd    | dddddddd | dddddddd   | ✅          | dddddddd   | ✅          |
```

### Observations:
- **Errors remain in four positions**, but they **shifted** in `v_scale=1.0` from epoch 10 to 20.
- The **same four inputs fail (`ac`, `bd`, `ca`, `db`)**, but their incorrect outputs changed.
- **The v_scale=1.0 errors clustered on extreme outputs (`aaaaaaaa`, `dddddddd`, `abababab`)**, whereas `v_scale=0.0` errors look more like misaligned cyclic mappings (`bcbcbcbc`, `babababa`, etc.).
- **Epoch 20 is worse than epoch 10**, meaning **training isn't refining these failure points**, possibly indicating **instability in how RoPE on `V` interacts with cyclic patterns**.

This suggests that applying RoPE to `V` isn't necessarily making attention more robust to cyclic structure—it's just *reorienting* failure points while keeping the same number of errors. The model could be overly weighting specific learned positional embeddings when `V` is modified.

### Next Steps:
- **Try intermediate `v_scale` values** (e.g., `0.3, 0.5, 0.7`) to see if errors gradually shift or disappear.
- **Analyze token embeddings at error positions**—is there a systematic drift in how they cluster?
- **Check attention weights on error cases**—do they collapse to single dominant tokens?

Would be great to visualize attention maps for `v_scale=0.0` vs. `v_scale=1.0` to see how it's warping representations!


HA! It was a problem in the dataset. It was missing those 4 definitions. The fact at lower epochs that it could even learn 2 is kinda amazing. Need to reproduce later.

So I can confirm now that 4 tokens in 2cw WORKS!

