from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import asyncio
import os
import tempfile
import aiofiles
from datetime import datetime

from models.base_model import AuthenticityClassifier
from training.trainer import RLTrainer
from database import db_manager
from config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Human Art Guardians ML Service",
    description="AI-powered content authenticity detection with reinforcement learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[AuthenticityClassifier] = None
trainer: Optional[RLTrainer] = None


# Pydantic models for API
class PredictionRequest(BaseModel):
    content_id: str
    content_url: str
    content_type: str


class PredictionResponse(BaseModel):
    content_id: str
    authentic_probability: float
    inauthentic_probability: float
    confidence: float
    prediction: str
    timestamp: str


class TrainingRequest(BaseModel):
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    content_items: List[PredictionRequest]


class FeedbackRequest(BaseModel):
    content_id: str
    user_id: str
    vote_type: str  # 'authentic_like' or 'inauthentic_dislike'
    prediction_accuracy: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model, trainer
    
    logger.info("Starting ML service...")
    
    try:
        # Initialize model
        model = AuthenticityClassifier()
        trainer = RLTrainer(model)
        
        # Try to load latest checkpoint
        checkpoint_dir = settings.CHECKPOINT_DIR
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                await trainer.load_checkpoint(checkpoint_path)
                logger.info(f"Loaded checkpoint: {latest_checkpoint}")
        
        logger.info("ML service initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing ML service: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "device": settings.DEVICE
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_authenticity(request: PredictionRequest) -> PredictionResponse:
    """
    Predict content authenticity for a single item
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        result = model.predict(request.content_url, request.content_type)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Update AI confidence in database
        await db_manager.update_ai_confidence(
            request.content_id, 
            result['confidence']
        )
        
        return PredictionResponse(
            content_id=request.content_id,
            authentic_probability=result['authentic_probability'],
            inauthentic_probability=result['inauthentic_probability'],
            confidence=result['confidence'],
            prediction=result['prediction'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error making prediction for {request.content_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest) -> List[PredictionResponse]:
    """
    Predict authenticity for multiple content items
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for item in request.content_items:
        try:
            prediction = model.predict(item.content_url, item.content_type)
            
            if 'error' not in prediction:
                # Update database
                await db_manager.update_ai_confidence(
                    item.content_id,
                    prediction['confidence']
                )
                
                results.append(PredictionResponse(
                    content_id=item.content_id,
                    authentic_probability=prediction['authentic_probability'],
                    inauthentic_probability=prediction['inauthentic_probability'],
                    confidence=prediction['confidence'],
                    prediction=prediction['prediction'],
                    timestamp=datetime.now().isoformat()
                ))
        except Exception as e:
            logger.warning(f"Error predicting {item.content_id}: {e}")
            continue
    
    return results


@app.post("/upload/predict")
async def predict_uploaded_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict authenticity for an uploaded file
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file size
    if len(await file.read()) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Reset file pointer
    await file.seek(0)
    
    # Determine content type
    content_type = "image"
    if file.content_type and file.content_type.startswith("video"):
        content_type = "video"
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Make prediction
        result = model.predict(tmp_path, content_type)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "filename": file.filename,
            "content_type": content_type,
            "authentic_probability": result['authentic_probability'],
            "inauthentic_probability": result['inauthentic_probability'],
            "confidence": result['confidence'],
            "prediction": result['prediction'],
            "timestamp": datetime.now().isoformat()
        }
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


@app.post("/feedback")
async def process_feedback(request: FeedbackRequest):
    """
    Process user feedback for online learning
    """
    try:
        # Get user voting patterns
        user_patterns = await db_manager.get_user_voting_patterns(request.user_id)
        
        # Store feedback for future training
        # This could trigger online learning updates
        
        return {
            "status": "success",
            "message": "Feedback processed",
            "user_patterns": user_patterns
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def start_training(background_tasks: BackgroundTasks, request: Optional[TrainingRequest] = None):
    """
    Start model training in the background
    """
    if not trainer:
        raise HTTPException(status_code=503, detail="Trainer not initialized")
    
    async def train_model():
        try:
            epochs = request.epochs if request and request.epochs else settings.MAX_EPOCHS
            await trainer.train(epochs)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    # Start training in background
    background_tasks.add_task(train_model)
    
    return {
        "status": "success",
        "message": "Training started in background",
        "epochs": request.epochs if request and request.epochs else settings.MAX_EPOCHS
    }


@app.get("/train/status")
async def get_training_status():
    """
    Get current training status
    """
    # This would need to be implemented with a proper task queue like Celery
    # For now, return a placeholder
    return {
        "status": "not_implemented",
        "message": "Training status tracking not yet implemented"
    }


@app.get("/analytics/summary")
async def get_analytics_summary():
    """
    Get analytics summary of model performance
    """
    try:
        # Get recent training data for analytics
        training_data = await db_manager.get_training_data(limit=100)
        
        total_content = len(training_data)
        
        # Calculate accuracy metrics
        authentic_count = sum(1 for item in training_data 
                            if item.get('authentic_votes_count', 0) > item.get('inauthentic_votes_count', 0))
        
        # Get recent votes for trending analysis
        recent_votes = await db_manager.get_recent_votes(hours=24)
        
        return {
            "total_content_analyzed": total_content,
            "authentic_content_ratio": authentic_count / max(total_content, 1),
            "recent_votes_24h": len(recent_votes),
            "model_confidence_avg": sum(item.get('ai_confidence_score', 0) for item in training_data) / max(total_content, 1),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the current model
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "device": settings.DEVICE,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
        "checkpoint_dir": settings.CHECKPOINT_DIR
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    ) 