import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str = "https://mhwtqawcuhxxcnlxgtdi.supabase.co"
    SUPABASE_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1od3RxYXdjdWh4eGNubHhndGRpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQwMzIyMzcsImV4cCI6MjA2OTYwODIzN30.HIXrM4g3QLnkNBh7ukFZiQVdpBLNh2p14kd_-n-oSi0"
    
    # Model Configuration
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    MODEL_CACHE_DIR: str = "./models"
    DEVICE: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # Training Configuration
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 16
    MAX_EPOCHS: int = 100
    PATIENCE: int = 10
    
    # Reinforcement Learning Configuration
    REWARD_SCALE: float = 1.0
    AUTHENTIC_REWARD: float = 1.0
    INAUTHENTIC_REWARD: float = -1.0
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Redis Configuration (for caching and task queue)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    WANDB_PROJECT: str = "human-art-guardians"
    
    # Model Paths
    CHECKPOINT_DIR: str = "./checkpoints"
    LOGS_DIR: str = "./logs"
    
    class Config:
        env_file = ".env"


settings = Settings() 