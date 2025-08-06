import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import logging
from config import settings

logger = logging.getLogger(__name__)


class AuthenticityClassifier(nn.Module):
    """
    Neural network for determining content authenticity
    Uses CLIP as feature extractor and adds classification head
    """
    
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        
        # Load pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=settings.MODEL_CACHE_DIR)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=settings.MODEL_CACHE_DIR)
        
        # Freeze CLIP parameters initially (can be unfrozen for fine-tuning)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Get CLIP vision model output dimension
        vision_embed_dim = self.clip_model.config.vision_config.hidden_size
        
        # Classification head for authenticity prediction
        self.classifier = nn.Sequential(
            nn.Linear(vision_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification: authentic vs inauthentic
        )
        
        # Value head for reinforcement learning
        self.value_head = nn.Sequential(
            nn.Linear(vision_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Value estimation
        )
        
        self.device = torch.device(settings.DEVICE)
        self.to(self.device)
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            images: Batch of preprocessed images
            
        Returns:
            Dictionary containing logits, probabilities, and value estimates
        """
        # Get vision features from CLIP
        vision_outputs = self.clip_model.vision_model(pixel_values=images)
        pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        
        # Classification logits
        logits = self.classifier(pooled_output)
        probabilities = F.softmax(logits, dim=-1)
        
        # Value estimation for RL
        values = self.value_head(pooled_output)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'values': values,
            'features': pooled_output
        }
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def preprocess_video(self, video_path: str, num_frames: int = 8) -> torch.Tensor:
        """Preprocess video by extracting key frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
            
            cap.release()
            
            if frames:
                inputs = self.clip_processor(images=frames, return_tensors="pt")
                return inputs['pixel_values'].to(self.device)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error preprocessing video {video_path}: {e}")
            return None
    
    def predict(self, content_path: str, content_type: str) -> Dict:
        """
        Make prediction on content
        
        Args:
            content_path: Path to content file
            content_type: 'image' or 'video'
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        
        with torch.no_grad():
            if content_type == 'image':
                inputs = self.preprocess_image(content_path)
            elif content_type == 'video':
                inputs = self.preprocess_video(content_path)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            if inputs is None:
                return {'error': 'Failed to preprocess content'}
            
            outputs = self.forward(inputs)
            
            # For video, aggregate predictions across frames
            if content_type == 'video' and inputs.shape[0] > 1:
                probs = outputs['probabilities'].mean(dim=0, keepdim=True)
                values = outputs['values'].mean(dim=0, keepdim=True)
            else:
                probs = outputs['probabilities']
                values = outputs['values']
            
            authentic_prob = probs[0, 1].item()  # Probability of being authentic
            confidence = max(probs[0]).item()    # Maximum probability
            value_estimate = values[0, 0].item() # Value estimation
            
            return {
                'authentic_probability': authentic_prob,
                'inauthentic_probability': 1 - authentic_prob,
                'confidence': confidence,
                'value_estimate': value_estimate,
                'prediction': 'authentic' if authentic_prob > 0.5 else 'inauthentic'
            }
    
    def unfreeze_clip(self):
        """Unfreeze CLIP parameters for fine-tuning"""
        for param in self.clip_model.parameters():
            param.requires_grad = True
    
    def freeze_clip(self):
        """Freeze CLIP parameters"""
        for param in self.clip_model.parameters():
            param.requires_grad = False


class RewardModel(nn.Module):
    """
    Reward model for reinforcement learning
    Predicts reward based on user feedback patterns
    """
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(feature_dim + 10, 256),  # +10 for user/content metadata
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Predict reward given features and metadata
        
        Args:
            features: Visual features from CLIP
            metadata: User and content metadata features
            
        Returns:
            Predicted reward
        """
        combined = torch.cat([features, metadata], dim=-1)
        return self.reward_predictor(combined) 