"""
Example script for training the minimal Transformer.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from fluctlight.model import MinimalTransformer
from fluctlight.dataset import Base64Dataset, create_dataloader

def main():
    # Data paths
    train_file = "data/train.txt"
    val_file = "data/val.txt"

    # Training parameters
    batch_size = 32
    max_epochs = 100
    learning_rate = 1e-3

    # Create datasets and dataloaders
    train_dataset = Base64Dataset(train_file)
    val_dataset = Base64Dataset(val_file)

    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size)

    # Create model
    model = MinimalTransformer(learning_rate=learning_rate)

    # Setup logger
    logger = TensorBoardLogger("lightning_logs", name="transformer")

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='transformer-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )

    # Create trainer with updated logging configuration
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator='auto',
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()