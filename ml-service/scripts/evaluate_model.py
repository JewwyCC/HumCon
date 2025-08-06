#!/usr/bin/env python3
"""
Model evaluation script for testing performance on validation data
Usage: python scripts/evaluate_model.py [--checkpoint PATH] [--output-report]
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import AuthenticityClassifier
from training.trainer import RLTrainer, ContentDataset
from database import db_manager
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model: AuthenticityClassifier):
        self.model = model
        self.evaluation_results = {}
    
    async def evaluate_on_dataset(self, dataset: ContentDataset) -> dict:
        """Evaluate model on a dataset"""
        logger.info(f"Evaluating model on {len(dataset)} samples...")
        
        predictions = []
        ground_truth = []
        confidence_scores = []
        content_types = []
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                content_id = sample['content_id']
                file_url = sample['file_url']
                content_type = sample['content_type']
                true_label = sample['label']
                
                if not file_url:
                    continue
                
                # Make prediction
                prediction = self.model.predict(file_url, content_type)
                
                if 'error' in prediction:
                    logger.warning(f"Error predicting {content_id}: {prediction['error']}")
                    continue
                
                predicted_label = 1 if prediction['authentic_probability'] > 0.5 else 0
                
                predictions.append(predicted_label)
                ground_truth.append(true_label)
                confidence_scores.append(prediction['confidence'])
                content_types.append(content_type)
                
                if i % 50 == 0:
                    logger.info(f"Processed {i}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        if not predictions:
            logger.error("No valid predictions made")
            return {}
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        
        # ROC AUC (if we have probability scores)
        try:
            authentic_probs = []
            for i in range(len(dataset)):
                sample = dataset[i]
                if sample['file_url']:
                    pred = self.model.predict(sample['file_url'], sample['content_type'])
                    if 'error' not in pred:
                        authentic_probs.append(pred['authentic_probability'])
                    else:
                        authentic_probs.append(0.5)  # Default
            
            if len(authentic_probs) == len(ground_truth):
                auc = roc_auc_score(ground_truth, authentic_probs)
            else:
                auc = 0.0
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc = 0.0
        
        # Per-content-type analysis
        content_type_metrics = {}
        for ct in set(content_types):
            ct_indices = [i for i, x in enumerate(content_types) if x == ct]
            if ct_indices:
                ct_gt = [ground_truth[i] for i in ct_indices]
                ct_pred = [predictions[i] for i in ct_indices]
                
                content_type_metrics[ct] = {
                    'accuracy': accuracy_score(ct_gt, ct_pred),
                    'count': len(ct_indices),
                    'authentic_ratio': sum(ct_gt) / len(ct_gt)
                }
        
        results = {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'num_samples': len(predictions)
            },
            'content_type_metrics': content_type_metrics,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'confidence_scores': confidence_scores,
            'content_types': content_types
        }
        
        return results
    
    def generate_plots(self, results: dict, output_dir: str):
        """Generate evaluation plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results['ground_truth'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Inauthentic', 'Authentic'],
                   yticklabels=['Inauthentic', 'Authentic'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confidence Distribution
        plt.figure(figsize=(10, 6))
        authentic_conf = [results['confidence_scores'][i] for i in range(len(results['ground_truth'])) 
                         if results['ground_truth'][i] == 1]
        inauthentic_conf = [results['confidence_scores'][i] for i in range(len(results['ground_truth'])) 
                           if results['ground_truth'][i] == 0]
        
        plt.hist(authentic_conf, alpha=0.7, label='Authentic', bins=30, color='green')
        plt.hist(inauthentic_conf, alpha=0.7, label='Inauthentic', bins=30, color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution by True Label')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Content Type Performance
        if results['content_type_metrics']:
            content_types = list(results['content_type_metrics'].keys())
            accuracies = [results['content_type_metrics'][ct]['accuracy'] for ct in content_types]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(content_types, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(content_types)])
            plt.ylabel('Accuracy')
            plt.title('Model Performance by Content Type')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'content_type_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}")
    
    def save_report(self, results: dict, output_path: str):
        """Save detailed evaluation report"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': settings.MODEL_NAME,
                'device': settings.DEVICE,
                'checkpoint_dir': settings.CHECKPOINT_DIR
            },
            'results': results,
            'summary': {
                'overall_accuracy': results['overall_metrics']['accuracy'],
                'total_samples': results['overall_metrics']['num_samples'],
                'content_types_tested': list(results['content_type_metrics'].keys()) if results['content_type_metrics'] else []
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description='Evaluate authenticity classification model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--test-size', type=int, default=500,
                       help='Number of test samples to use')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    logger.info("Initializing model for evaluation...")
    
    # Initialize model
    model = AuthenticityClassifier()
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer = RLTrainer(model)
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        await trainer.load_checkpoint(args.checkpoint)
    else:
        logger.info("Using model without checkpoint (random weights)")
    
    # Prepare evaluation data
    logger.info("Preparing evaluation dataset...")
    raw_data = await db_manager.get_training_data(limit=args.test_size)
    
    if not raw_data:
        logger.error("No data available for evaluation")
        return
    
    # Filter data with sufficient votes
    test_data = [
        item for item in raw_data 
        if (item.get('authentic_votes_count', 0) + item.get('inauthentic_votes_count', 0)) >= 3
    ]
    
    if not test_data:
        logger.error("No data with sufficient votes for evaluation")
        return
    
    logger.info(f"Using {len(test_data)} samples for evaluation")
    
    test_dataset = ContentDataset(test_data, is_training=False)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Run evaluation
    logger.info("Running model evaluation...")
    results = await evaluator.evaluate_on_dataset(test_dataset)
    
    if not results:
        logger.error("Evaluation failed - no results generated")
        return
    
    # Print summary
    metrics = results['overall_metrics']
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"  AUC: {metrics['auc']:.4f}")
    logger.info(f"  Samples: {metrics['num_samples']}")
    
    # Content type breakdown
    if results['content_type_metrics']:
        logger.info("\nContent Type Performance:")
        for content_type, ct_metrics in results['content_type_metrics'].items():
            logger.info(f"  {content_type}: {ct_metrics['accuracy']:.4f} ({ct_metrics['count']} samples)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save report
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    evaluator.save_report(results, report_path)
    
    # Generate plots
    if args.generate_plots:
        logger.info("Generating evaluation plots...")
        evaluator.generate_plots(results, args.output_dir)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 