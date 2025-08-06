import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import wandb
from tqdm import tqdm
from datetime import datetime
import os
import json

from models.base_model import AuthenticityClassifier, RewardModel
from database import db_manager
from config import settings

logger = logging.getLogger(__name__)


class ContentDataset(Dataset):
    """Dataset for content authenticity training"""
    
    def __init__(self, data: List[Dict], is_training: bool = True):
        self.data = data
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Calculate labels from votes
        authentic_votes = item.get('authentic_votes_count', 0)
        inauthentic_votes = item.get('inauthentic_votes_count', 0)
        total_votes = authentic_votes + inauthentic_votes
        
        if total_votes == 0:
            # No votes yet, skip or use prior
            label = 0.5  # Neutral
            confidence = 0.0
        else:
            label = authentic_votes / total_votes
            confidence = min(total_votes / 10.0, 1.0)  # Confidence based on vote count
        
        # Binary label for classification
        binary_label = 1 if label > 0.5 else 0
        
        return {
            'content_id': item['id'],
            'file_url': item.get('file_url', ''),
            'content_type': item.get('content_type', 'image'),
            'label': binary_label,
            'confidence': confidence,
            'authentic_ratio': label,
            'total_votes': total_votes,
            'metadata': {
                'creator_id': item.get('creator_id', ''),
                'created_at': item.get('created_at', ''),
                'ai_confidence_score': item.get('ai_confidence_score', 0.0)
            }
        }


class RLTrainer:
    """
    Reinforcement Learning trainer for authenticity classification
    Uses user feedback as rewards to improve model performance
    """
    
    def __init__(self, model: AuthenticityClassifier):
        self.model = model
        self.reward_model = RewardModel()
        self.device = torch.device(settings.DEVICE)
        
        # Optimizers
        self.model_optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=settings.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        self.reward_optimizer = optim.AdamW(
            self.reward_model.parameters(),
            lr=settings.LEARNING_RATE * 0.1
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        self.reward_loss = nn.MSELoss()
        
        # Training history
        self.training_history = []
        
        # Initialize wandb if configured
        if settings.WANDB_PROJECT:
            wandb.init(project=settings.WANDB_PROJECT, config=settings.dict())
    
    async def prepare_training_data(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        logger.info("Fetching training data from database...")
        
        # Get all content with votes
        raw_data = await db_manager.get_training_data(limit=5000)
        
        # Filter content with sufficient votes for reliable labels
        filtered_data = [
            item for item in raw_data 
            if (item.get('authentic_votes_count', 0) + item.get('inauthentic_votes_count', 0)) >= 3
        ]
        
        logger.info(f"Loaded {len(filtered_data)} content items for training")
        
        # Split data
        split_idx = int(0.8 * len(filtered_data))
        train_data = filtered_data[:split_idx]
        val_data = filtered_data[split_idx:]
        
        train_dataset = ContentDataset(train_data, is_training=True)
        val_dataset = ContentDataset(val_data, is_training=False)
        
        return train_dataset, val_dataset
    
    def compute_reward(self, prediction: Dict, actual_votes: Dict, user_patterns: Dict) -> float:
        """
        Compute reward based on prediction accuracy and user feedback
        
        Args:
            prediction: Model prediction results
            actual_votes: Actual vote counts
            user_patterns: User voting patterns for personalization
            
        Returns:
            Computed reward value
        """
        authentic_votes = actual_votes.get('authentic_votes_count', 0)
        inauthentic_votes = actual_votes.get('inauthentic_votes_count', 0)
        total_votes = authentic_votes + inauthentic_votes
        
        if total_votes == 0:
            return 0.0
        
        # Ground truth from majority vote
        true_authentic_ratio = authentic_votes / total_votes
        predicted_authentic_prob = prediction.get('authentic_probability', 0.5)
        
        # Base reward: accuracy of prediction
        base_reward = 1.0 - abs(true_authentic_ratio - predicted_authentic_prob)
        
        # Confidence bonus: reward high confidence when correct
        confidence = prediction.get('confidence', 0.5)
        if (true_authentic_ratio > 0.5 and predicted_authentic_prob > 0.5) or \
           (true_authentic_ratio <= 0.5 and predicted_authentic_prob <= 0.5):
            confidence_bonus = confidence * 0.5
        else:
            confidence_bonus = -confidence * 0.5
        
        # Vote count bonus: more reliable when more votes
        vote_weight = min(total_votes / 10.0, 1.0)
        
        # User pattern adjustment (optional personalization)
        user_agreement = user_patterns.get('authentic_ratio', 0.5)
        if abs(user_agreement - predicted_authentic_prob) < 0.3:
            user_bonus = 0.1
        else:
            user_bonus = 0.0
        
        total_reward = (base_reward + confidence_bonus + user_bonus) * vote_weight
        return np.clip(total_reward, -2.0, 2.0)
    
    async def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch in pbar:
            self.model_optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_reward = 0.0
            batch_size = len(batch['content_id'])
            
            for i in range(batch_size):
                content_id = batch['content_id'][i]
                file_url = batch['file_url'][i]
                content_type = batch['content_type'][i]
                true_label = batch['label'][i].item()
                
                if not file_url:
                    continue
                
                try:
                    # Make prediction
                    prediction = self.model.predict(file_url, content_type)
                    
                    if 'error' in prediction:
                        continue
                    
                    # Get actual votes and user patterns
                    content_data = await db_manager.get_content_by_id(content_id)
                    if not content_data:
                        continue
                    
                    # Compute reward
                    reward = self.compute_reward(
                        prediction, 
                        content_data, 
                        {}  # Could add user patterns here
                    )
                    
                    # Convert prediction to tensor for loss computation
                    pred_tensor = torch.tensor([
                        1 - prediction['authentic_probability'],
                        prediction['authentic_probability']
                    ]).unsqueeze(0).to(self.device)
                    
                    true_tensor = torch.tensor([true_label]).long().to(self.device)
                    
                    # Classification loss
                    cls_loss = self.classification_loss(pred_tensor, true_tensor)
                    
                    # RL loss: encourage actions that lead to higher rewards
                    value_target = torch.tensor([reward]).float().to(self.device)
                    
                    # Get model features for value estimation
                    if content_type == 'image':
                        inputs = self.model.preprocess_image(file_url)
                    else:
                        inputs = self.model.preprocess_video(file_url)
                    
                    if inputs is not None:
                        outputs = self.model.forward(inputs)
                        value_loss = self.value_loss(outputs['values'], value_target.unsqueeze(0))
                        
                        # Total loss
                        loss = cls_loss + 0.5 * value_loss + 0.1 * torch.abs(torch.tensor(reward))
                        batch_loss += loss.item()
                        batch_reward += reward
                        
                        # Backward pass
                        loss.backward()
                    
                except Exception as e:
                    logger.warning(f"Error processing content {content_id}: {e}")
                    continue
            
            # Update model
            if batch_loss > 0:
                self.model_optimizer.step()
                
                total_loss += batch_loss
                total_reward += batch_reward
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{batch_loss:.4f}",
                    'Reward': f"{batch_reward:.4f}"
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_reward = total_reward / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'reward': avg_reward,
            'accuracy': total_accuracy / max(num_batches, 1)
        }
    
    async def validate(self, val_loader: DataLoader) -> Dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_reward = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_size = len(batch['content_id'])
                
                for i in range(batch_size):
                    content_id = batch['content_id'][i]
                    file_url = batch['file_url'][i]
                    content_type = batch['content_type'][i]
                    true_label = batch['label'][i].item()
                    
                    if not file_url:
                        continue
                    
                    try:
                        prediction = self.model.predict(file_url, content_type)
                        
                        if 'error' in prediction:
                            continue
                        
                        predicted_label = 1 if prediction['authentic_probability'] > 0.5 else 0
                        accuracy = 1.0 if predicted_label == true_label else 0.0
                        
                        total_accuracy += accuracy
                        num_samples += 1
                        
                    except Exception as e:
                        continue
        
        avg_accuracy = total_accuracy / max(num_samples, 1)
        
        return {
            'accuracy': avg_accuracy,
            'num_samples': num_samples
        }
    
    async def train(self, num_epochs: int = None) -> None:
        """Main training loop"""
        if num_epochs is None:
            num_epochs = settings.MAX_EPOCHS
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Prepare data
        train_dataset, val_dataset = await self.prepare_training_data()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=settings.BATCH_SIZE, 
            shuffle=True,
            collate_fn=lambda x: {key: [item[key] for item in x] for key in x[0].keys()}
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=settings.BATCH_SIZE, 
            shuffle=False,
            collate_fn=lambda x: {key: [item[key] for item in x] for key in x[0].keys()}
        )
        
        best_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = await self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = await self.validate(val_loader)
            
            # Logging
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Reward: {train_metrics['reward']:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )
            
            # Wandb logging
            if settings.WANDB_PROJECT:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_reward': train_metrics['reward'],
                    'val_accuracy': val_metrics['accuracy']
                })
            
            # Save best model
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                patience_counter = 0
                await self.save_checkpoint(epoch, val_metrics['accuracy'])
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= settings.PATIENCE:
                logger.info(f"Early stopping after {epoch} epochs")
                break
        
        logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    
    async def save_checkpoint(self, epoch: int, accuracy: float) -> None:
        """Save model checkpoint"""
        os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model_optimizer.state_dict(),
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            settings.CHECKPOINT_DIR, 
            f"model_checkpoint_epoch_{epoch}_acc_{accuracy:.4f}.pt"
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}") 