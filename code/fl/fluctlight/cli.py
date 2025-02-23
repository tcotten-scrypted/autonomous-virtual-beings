"""
Command-line interface for training and using the minimal Transformer.

This module provides a CLI for:
1. Training the model on Base64-encoded text data
2. Generating text continuations using a trained model
3. Managing checkpoints and logging

The interface supports both training new models and using pre-trained models
for text generation, with configurable parameters for both processes.
"""

import argparse
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .model import MinimalTransformer
from .dataset import Base64Dataset, create_dataloader
from .utils import generate_continuation

def train(
    train_file: str,
    val_file: str,
    output_dir: str,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3
):
    """
    Train the model on Base64-encoded text data.

    The training process includes:
    - Loading training and validation datasets
    - Setting up TensorBoard logging
    - Configuring checkpointing
    - Training the model with specified parameters

    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        output_dir: Directory to save checkpoints
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data
    train_dataset = Base64Dataset(train_file)
    val_dataset = Base64Dataset(val_file)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size)

    # Create model
    model = MinimalTransformer(learning_rate=learning_rate)

    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="transformer")

    # Setup training
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='auto',
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

def generate(
    checkpoint_path: str,
    input_text: str,
    max_length: int = 100,
    temperature: float = 1.0
):
    """
    Generate text continuation using a trained model.

    The generation process:
    1. Loads a trained model from checkpoint
    2. Takes input text as seed
    3. Generates continuation with specified parameters
    4. Prints both input and generated text

    Args:
        checkpoint_path: Path to model checkpoint
        input_text: Seed text for generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
    """
    # Load model
    model = MinimalTransformer.load_from_checkpoint(checkpoint_path)

    # Generate continuation
    output = generate_continuation(
        model,
        input_text,
        max_length=max_length,
        temperature=temperature
    )

    print(f"Input: {input_text}")
    print(f"Generated: {output}")

def main():
    parser = argparse.ArgumentParser(description="Minimal Transformer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train-file", required=True, help="Training data file")
    train_parser.add_argument("--val-file", required=True, help="Validation data file")
    train_parser.add_argument("--output-dir", required=True, help="Output directory")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    generate_parser.add_argument("--input-text", required=True, help="Input text")
    generate_parser.add_argument("--max-length", type=int, default=100, help="Maximum length")
    generate_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    if args.command == "train":
        train(
            args.train_file,
            args.val_file,
            args.output_dir,
            args.batch_size,
            args.max_epochs,
            args.learning_rate
        )
    elif args.command == "generate":
        generate(
            args.checkpoint,
            args.input_text,
            args.max_length,
            args.temperature
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()