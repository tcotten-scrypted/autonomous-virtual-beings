"""
Command-line interface for training and using the Fluctlight Transformer.

This module provides a CLI for:
1. Training the model on Base64-encoded text data
2. Generating text continuations using a trained model
3. Testing model predictions against expected outputs
4. Managing checkpoints and logging

The interface is optimized for the minimal context window of 2 tokens,
which is sufficient for learning basic patterns. The training process
includes ASCII cycle data to prevent mode collapse around Token 0.

Examples:
    Training:
        python -m fluctlight.cli train \\
            --train-file data/train.txt \\
            --val-file data/val.txt \\
            --output-dir checkpoints \\
            --batch-size 32
            
    Generation:
        python -m fluctlight.cli generate \\
            --checkpoint checkpoints/last.ckpt \\
            --input-text "ab" \\
            --temperature 0.2  # Low temp for stable patterns
            
    Testing:
        python -m fluctlight.cli test \\
            --checkpoint checkpoints/last.ckpt \\
            --input-file tests/patterns.csv \\
            --temperature 0.01  # Very low temp for deterministic output
"""

import argparse
import base64
import csv
import os
import sys
from pathlib import Path
from typing import Optional, List, Iterator, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Handle both module and direct script usage
try:
    from .model import FluctlightTransformer
    from .dataset import Base64Dataset, create_dataloader
    from .utils import generate_continuation, load_model, calculate_rmse
except ImportError:
    # Add parent directory to path for direct script usage
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fluctlight.model import FluctlightTransformer
    from fluctlight.dataset import Base64Dataset, create_dataloader
    from fluctlight.utils import generate_continuation, load_model, calculate_rmse

def ascii_chunks(size: int = 32) -> Iterator[bytes]:
    """
    Generate overlapping chunks of the ASCII character set (0-255).
    
    Creates chunks that wrap around at the end to form a cycle,
    ensuring all byte values are represented in training data.
    
    Args:
        size: Size of each chunk (default: 32)
        
    Returns:
        Iterator[bytes]: Overlapping byte chunks
    """
    ascii_bytes = bytes(range(256))  # All byte values from 0 to 255
    chunks = [ascii_bytes[i:i + size] for i in range(0, 256, size)]

    # Ensure wrap-around by appending the first chunk at the end
    chunks.append(ascii_bytes[:size])
    return chunks

def generate_seed_data() -> List[str]:
    """
    Generate cyclically mapped Base64 seed data.
    
    Creates training data that helps prevent mode collapse by ensuring
    all byte values are represented in the training set. This is
    particularly important for the minimal context window (2 tokens).
    
    Format:
        base64(32-byte chunk) \t base64(next 32-byte chunk) \n
        
    Returns:
        List[str]: Base64-encoded training pairs
    """
    chunks = ascii_chunks()
    data = []
    
    for i in range(len(chunks) - 1):
        left = base64.b64encode(chunks[i]).decode('utf-8')
        right = base64.b64encode(chunks[i + 1]).decode('utf-8')
        data.append(f"{left}\t{right}\n")

    return data

class OverwriteLastCheckpoint(pl.Callback):
    """Callback to save the latest model state after each validation."""
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize the callback.
        
        Args:
            output_dir: Directory to save checkpoints
        """
        super().__init__()
        self.dirpath = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save checkpoint after validation."""
        checkpoint_path = os.path.join(self.dirpath, 'last.ckpt')
        trainer.save_checkpoint(checkpoint_path)

def train(
    train_file: str,
    val_file: str,
    output_dir: str,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    gradient_clip_val: float = 1.0,
    vocab_size: int = 256,
    context_window: Optional[int] = None,
    v_scale: float = 0.0
) -> None:
    """
    Train the Fluctlight model on Base64-encoded text data.

    The training process includes:
    1. Loading training and validation datasets
    2. Adding ASCII cycle data to prevent mode collapse
    3. Setting up TensorBoard logging
    4. Configuring checkpointing
    5. Training with specified parameters

    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        output_dir: Directory to save checkpoints
        batch_size: Training batch size (default: 32)
        max_epochs: Maximum training epochs (default: 100)
        learning_rate: Initial learning rate (default: 1e-3)
        gradient_clip_val: Gradient clipping value (default: 1.0)
        vocab_size: Size of vocabulary (default: 256 for bytes)
        context_window: Size of context window (default: None, will be predicted)
        v_scale: Scale factor for RoPE on value vectors (default: 0.0)
        
    Raises:
        FileNotFoundError: If data files don't exist
        ValueError: If parameters are invalid
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = FluctlightTransformer(
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        context_window=context_window,
        v_scale=v_scale
    )

    # Prepare data
    train_dataset = Base64Dataset(train_file, prepend=generate_seed_data())
    val_dataset = Base64Dataset(val_file)

    train_loader = create_dataloader(train_dataset, model.context_window, batch_size=batch_size)
    val_loader = create_dataloader(val_dataset, model.context_window, batch_size=batch_size)

    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="transformer")

    # Setup training
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss'
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, OverwriteLastCheckpoint(output_dir)],
        logger=logger,
        accelerator='auto',
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        gradient_clip_val=gradient_clip_val
    )

    trainer.fit(model, train_loader, val_loader)

def generate(
    checkpoint_path: str,
    input_text: str,
    max_length: Optional[int] = None,
    temperature: float = 0.8
) -> None:
    """
    Generate text continuation using a trained model.

    The generation process:
    1. Loads a trained model from checkpoint
    2. Takes input text as seed
    3. Generates continuation with specified parameters
    4. Prints both input and generated text

    For the minimal context window (2 tokens), temperature control
    is crucial:
    - 0.1-0.3: Stable patterns (e.g., "ababab")
    - 0.4-0.7: Balanced variation
    - 0.8-1.0: Maximum variation

    Args:
        checkpoint_path: Path to model checkpoint
        input_text: Seed text for generation
        max_length: Maximum length to generate
        temperature: Sampling temperature (default: 0.8)
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If temperature <= 0
    """
    # Load model
    model = FluctlightTransformer.load_from_checkpoint(checkpoint_path)

    # Generate continuation
    output = generate_continuation(
        model,
        input_text,
        max_length=max_length,
        temperature=temperature
    )

    print(f"Input: {input_text}")
    print(f"Generated: {output}")

def test(
    checkpoint_path: str,
    input_file: str,
    temperature: float = 0.01,
    debugging: bool = False
) -> None:
    """
    Test model predictions against expected outputs from a CSV file.
    
    The CSV file should contain two columns:
    1. Input tokens (e.g., "ab")
    2. Expected output (e.g., "abababab")
    
    The function prints a table with:
    - ✅/❌ for exact matches
    - Number of errors
    - RMSE between expected and actual outputs
    - Input tokens
    - Expected output
    - Actual output
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_file: Path to CSV file with test cases
        temperature: Sampling temperature (default: 0.01 for deterministic output)
        debugging: Whether to print detailed model information (default: False)
        
    Raises:
        FileNotFoundError: If checkpoint or input file doesn't exist
        ValueError: If temperature <= 0 or CSV format is invalid
    """
    # Load model once for all test cases
    model = load_model(checkpoint_path)
    
    if debugging:
        print("\n=== Model Configuration ===")
        print(f"Architecture:")
        print(f"  - Embedding Dimension (d_model): {model.d_model}")
        print(f"  - Number of Heads: {model.n_heads}")
        print(f"  - Number of Layers: {model.n_layers}")
        print(f"  - Head Dimension: {model.head_dim}")
        print(f"  - Feed-forward Dimension: {model.d_ff}")
        print(f"\nParameters:")
        print(f"  - Vocabulary Size: {model.vocab_size}")
        print(f"  - Context Window: {model.context_window}")
        print(f"  - RoPE V-scale: {model.v_scale}")
        print(f"  - Dropout Rate: {model.dropout_rate}")
        print(f"\nGeneration Settings:")
        print(f"  - Temperature: {temperature}")
        print(f"  - Device: {model.device}")
        print("\nTest Configuration:")
        print(f"  - Input File: {input_file}")
        print(f"  - Checkpoint: {checkpoint_path}")
        print("\n=== Test Results ===\n")
    
    # Print table header
    print("match,errors,rmse,input,expected,actual")
    
    try:
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 2:
                    raise ValueError(f"Invalid CSV format. Expected 2 columns, got {len(row)}")
                
                input_str, expected = row
                
                # Generate output using the loaded model
                actual = generate_continuation(
                    model,
                    input_str,
                    max_length=len(expected),  # Match expected length
                    temperature=temperature
                )
                
                # Calculate metrics
                is_match = actual == expected
                error_count = sum(1 for a, e in zip(actual.ljust(len(expected)), expected) if a != e)
                rmse = calculate_rmse(expected, actual)
                
                # Print result row
                print(f"{('✅' if is_match else '❌')},{error_count},{rmse:.3f},{input_str},{expected},{actual}")
                
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input file not found: {input_file}") from e
    except csv.Error as e:
        raise ValueError(f"Invalid CSV format: {e}") from e

def main() -> None:
    """
    Main entry point for the CLI.
    
    Provides commands:
    1. train: Train a new model
    2. generate: Generate text with a trained model
    3. test: Test model predictions against expected outputs
    
    Run with --help for usage information.
    """
    parser = argparse.ArgumentParser(description="Fluctlight Transformer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train-file", required=True, help="Training data file")
    train_parser.add_argument("--val-file", required=True, help="Validation data file")
    train_parser.add_argument("--output-dir", required=True, help="Output directory")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value")
    train_parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    train_parser.add_argument("--context-window", type=int, default=None, help="Context window size (default: auto-predict)")
    train_parser.add_argument("--v-scale", type=float, default=0.0, help="RoPE scale factor for value vectors (default: 0.0)")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    generate_parser.add_argument("--input-text", required=True, help="Input text")
    generate_parser.add_argument("--max-length", type=int, default=None, help="Maximum length")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test model predictions")
    test_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    test_parser.add_argument("--input-file", required=True, help="CSV file with test cases")
    test_parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature (default: 0.01)")
    test_parser.add_argument("--debugging", action="store_true", help="Print detailed model information and generation process")

    args = parser.parse_args()

    if args.command == "train":
        train(
            args.train_file,
            args.val_file,
            args.output_dir,
            args.batch_size,
            args.max_epochs,
            args.learning_rate,
            args.gradient_clip_val,
            args.vocab_size,
            args.context_window,
            args.v_scale
        )
    elif args.command == "generate":
        generate(
            args.checkpoint,
            args.input_text,
            args.max_length,
            args.temperature
        )
    elif args.command == "test":
        test(
            args.checkpoint,
            args.input_file,
            args.temperature,
            args.debugging
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
