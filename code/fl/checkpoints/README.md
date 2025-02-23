# Model Checkpoints Directory

This directory stores model checkpoints generated during training. The checkpoints are not tracked in git, but will be created when you run the training script.

Checkpoint files follow the naming pattern:
```
transformer-{epoch:02d}-{val_loss:.2f}.ckpt
```

For example: `transformer-59-1.78.ckpt`
