#!/usr/bin/env python3
"""
Standalone training script for the authenticity classification model
Usage: python scripts/train_model.py [--epochs N] [--lr RATE] [--batch-size N]
"""

import asyncio
import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import AuthenticityClassifier
from training.trainer import RLTrainer
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description='Train authenticity classification model')
    parser.add_argument('--epochs', type=int, default=settings.MAX_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=settings.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=settings.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--unfreeze-clip', action='store_true',
                       help='Unfreeze CLIP parameters for fine-tuning')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, do not train')
    
    args = parser.parse_args()
    
    # Update settings with command line arguments
    settings.MAX_EPOCHS = args.epochs
    settings.LEARNING_RATE = args.lr
    settings.BATCH_SIZE = args.batch_size
    
    logger.info("Initializing model and trainer...")
    
    # Initialize model
    model = AuthenticityClassifier()
    
    if args.unfreeze_clip:
        logger.info("Unfreezing CLIP parameters for fine-tuning")
        model.unfreeze_clip()
    else:
        logger.info("Keeping CLIP parameters frozen")
    
    # Initialize trainer
    trainer = RLTrainer(model)
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        await trainer.load_checkpoint(args.checkpoint)
    
    # Create necessary directories
    os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    
    if args.validate_only:
        logger.info("Running validation only...")
        
        # Prepare validation data
        _, val_dataset = await trainer.prepare_training_data()
        
        from torch.utils.data import DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=settings.BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda x: {key: [item[key] for item in x] for key in x[0].keys()}
        )
        
        # Run validation
        val_metrics = await trainer.validate(val_loader)
        
        logger.info(f"Validation Results:")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Samples: {val_metrics['num_samples']}")
        
    else:
        logger.info(f"Starting training for {args.epochs} epochs...")
        logger.info(f"Configuration:")
        logger.info(f"  Learning Rate: {settings.LEARNING_RATE}")
        logger.info(f"  Batch Size: {settings.BATCH_SIZE}")
        logger.info(f"  Device: {settings.DEVICE}")
        logger.info(f"  Model: {settings.MODEL_NAME}")
        
        # Start training
        try:
            await trainer.train(args.epochs)
            logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main()) 