# model_inspector.py
import torch
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from fluctlight.model import FluctlightTransformer

def inspect_model(checkpoint_path):
    model = FluctlightTransformer.load_from_checkpoint(checkpoint_path)
    
    print(f"===== Model Parameters =====")
    print(f"Context window: {model.context_window}")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Embedding dimension (d_model): {model.d_model}")
    print(f"Number of heads: {model.n_heads}")
    print(f"Number of layers: {model.n_layers}")
    
    # Access hyperparameters dict for d_ff instead of direct attribute
    if hasattr(model, 'hparams') and 'd_ff' in model.hparams:
        print(f"Feed-forward dimension: {model.hparams.d_ff}")
    
    print(f"Learning rate: {model.learning_rate}")
    print(f"Weight decay: {model.weight_decay}")
    
    # Access dropout_rate safely
    if hasattr(model, 'dropout_rate'):
        print(f"Dropout rate: {model.dropout_rate}")
    
    # Print all hyperparameters
    print("\nAll hyperparameters:")
    if hasattr(model, 'hparams'):
        for key, value in model.hparams.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_inspector.py <checkpoint_path>")
        sys.exit(1)
        
    inspect_model(sys.argv[1])
