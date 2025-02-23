# Training Data Directory

This directory stores training and validation data files used by the model. These files are not tracked in git, but will be created when you prepare your training data.

Sample files:
```
sample-train.txt # Training data based on @elonmusk X posts
sample-val.txt # Validations based on @elonmusk X posts
```

User expected files:
```
train.txt  # Training data file (Base64 encoded input-output pairs)
val.txt    # Validation data file (Base64 encoded input-output pairs)
```

## File Format
Each line in these files should be a Base64-encoded string representing an input-output pair separated by a tab character. For example:
```
SGVsbG8=\tV29ybGQ= # Hello World
```