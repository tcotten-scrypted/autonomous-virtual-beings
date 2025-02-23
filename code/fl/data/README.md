# Training Data Directory

This directory stores training and validation data files used by the model. These files are not tracked in git, but will be created when you prepare your training data.

Expected files:
```
train.txt  # Training data file (Base64 encoded input-output pairs)
val.txt    # Validation data file (Base64 encoded input-output pairs)
```

## File Format
Each line in these files should be a Base64-encoded string representing an input-output pair separated by a tab character. For example:
```
SGVsbG8JV29ybGQ=  # Decodes to: "Hello\tWorld"
```
