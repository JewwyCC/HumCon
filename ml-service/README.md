# Human Art Guardians - ML Service

An AI-powered content authenticity detection system with reinforcement learning capabilities, designed to identify authentic vs. AI-generated content based on user feedback.

## 🚀 Features

- **CLIP-based Visual Recognition**: Uses OpenAI's CLIP model for robust image and video analysis
- **Reinforcement Learning**: Continuously improves based on user feedback (upvotes/downvotes)
- **Real-time Inference**: Fast API endpoints for content analysis
- **Batch Processing**: Efficient analysis of multiple content items
- **Supabase Integration**: Seamless connection with existing database
- **Model Training Pipeline**: Automated training with user feedback data
- **Analytics Dashboard**: Comprehensive metrics and monitoring
- **Docker Support**: Easy deployment and scaling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React App     │    │   ML Service    │    │   Supabase DB   │
│                 │◄──►│                 │◄──►│                 │
│ - Upload UI     │    │ - CLIP Model    │    │ - Content       │
│ - Analytics     │    │ - RL Training   │    │ - Votes         │
│ - Feedback      │    │ - FastAPI       │    │ - Users         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for faster training)
- Redis (for caching and task queues)
- 8GB+ RAM recommended
- ~2GB storage for models

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd human-art-guardians/ml-service
```

### 2. Create Virtual Environment

```bash
python -m venv ml-env
source ml-env/bin/activate  # On Windows: ml-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# Model Configuration
MODEL_NAME=openai/clip-vit-base-patch32
DEVICE=cuda  # or cpu
BATCH_SIZE=16

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Optional: Weights & Biases for experiment tracking
WANDB_PROJECT=human-art-guardians
WANDB_API_KEY=your_wandb_key
```

### 5. Start Redis (if not using Docker)

```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

## 🚀 Quick Start

### Method 1: Direct Python

```bash
# Start the ML service
python api/main.py

# In another terminal, run initial training
python scripts/train_model.py --epochs 5
```

### Method 2: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Check logs
docker-compose logs -f ml-service
```

### Method 3: Manual Docker

```bash
# Build image
docker build -t human-art-guardians-ml .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  human-art-guardians-ml
```

## 📚 API Documentation

Once running, visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Key Endpoints

#### Content Analysis
```bash
# Predict single content
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "123",
    "content_url": "https://example.com/image.jpg",
    "content_type": "image"
  }'

# Upload and analyze file
curl -X POST "http://localhost:8000/upload/predict" \
  -F "file=@/path/to/image.jpg"
```

#### Training & Feedback
```bash
# Send user feedback
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "123",
    "user_id": "user456",
    "vote_type": "authentic_like"
  }'

# Start training
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10}'
```

## 🔧 Training the Model

### Initial Training

```bash
# Basic training (10 epochs)
python scripts/train_model.py

# Advanced training options
python scripts/train_model.py \
  --epochs 50 \
  --lr 1e-4 \
  --batch-size 32 \
  --unfreeze-clip  # Fine-tune CLIP weights
```

### Continuing from Checkpoint

```bash
python scripts/train_model.py \
  --checkpoint ./checkpoints/model_checkpoint_epoch_10_acc_0.85.pt \
  --epochs 20
```

### Evaluation

```bash
# Evaluate model performance
python scripts/evaluate_model.py \
  --checkpoint ./checkpoints/best_model.pt \
  --generate-plots \
  --output-dir ./evaluation_results
```

## 📊 Model Performance

The system uses several metrics to evaluate performance:

- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Per-class performance
- **AUC-ROC**: Area under the ROC curve
- **Confidence**: Model prediction confidence
- **User Agreement**: How well predictions align with user votes

### Expected Performance

| Content Type | Accuracy | Confidence |
|-------------|----------|------------|
| Images      | 75-85%   | 80-90%     |
| Videos      | 70-80%   | 75-85%     |
| Overall     | 73-83%   | 78-88%     |

*Performance improves with more training data and user feedback*

## 🔄 Reinforcement Learning

The system implements a custom RL algorithm:

1. **Reward Calculation**: Based on user vote accuracy
2. **Value Function**: Estimates expected future rewards
3. **Policy Updates**: Adjusts model based on feedback
4. **Online Learning**: Continuous improvement from new votes

### Reward Function

```python
def compute_reward(prediction, actual_votes, user_patterns):
    # Base reward: prediction accuracy
    accuracy_reward = 1.0 - abs(predicted_prob - actual_ratio)
    
    # Confidence bonus: reward high confidence when correct
    confidence_bonus = confidence * 0.5 if correct else -confidence * 0.5
    
    # Vote weight: more reliable with more votes
    vote_weight = min(total_votes / 10.0, 1.0)
    
    return (accuracy_reward + confidence_bonus) * vote_weight
```

## 🔗 Frontend Integration

Add to your React app:

```typescript
// Install the ML service integration
import { mlService } from '@/services/mlService';
import { useMLAnalytics } from '@/hooks/useMLAnalytics';

// Use in components
const { predictContent, analytics } = useMLAnalytics();

// Predict content authenticity
const prediction = await predictContent(
  contentId, 
  contentUrl, 
  contentType
);
```

## 📈 Monitoring & Analytics

### Built-in Analytics

- Content analysis metrics
- Model performance tracking
- User voting patterns
- System health monitoring

### External Monitoring

- **Weights & Biases**: Experiment tracking
- **Redis Commander**: Cache monitoring
- **Docker logs**: System logs
- **FastAPI metrics**: API performance

## 🔧 Configuration

### Model Parameters

```python
# config.py
LEARNING_RATE = 1e-4      # Training learning rate
BATCH_SIZE = 16           # Training batch size
MAX_EPOCHS = 100          # Maximum training epochs
PATIENCE = 10             # Early stopping patience

# Reinforcement Learning
AUTHENTIC_REWARD = 1.0    # Reward for correct authentic prediction
INAUTHENTIC_REWARD = -1.0 # Penalty for incorrect prediction
CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence for high-confidence predictions
```

### Hardware Optimization

```bash
# For CPU-only systems
export DEVICE=cpu

# For GPU systems
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

# For multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

## 🚨 Troubleshooting

### Common Issues

1. **Model not loading**:
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Force CPU mode
   export DEVICE=cpu
   ```

2. **Out of memory errors**:
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   
   # Use smaller model
   export MODEL_NAME=openai/clip-vit-base-patch16
   ```

3. **Database connection issues**:
   ```bash
   # Test Supabase connection
   python -c "from database import db_manager; print('Connected!' if db_manager.supabase else 'Failed')"
   ```

4. **Redis connection issues**:
   ```bash
   # Test Redis connection
   redis-cli ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

### Performance Optimization

1. **Enable GPU acceleration**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Optimize batch size**:
   ```python
   # Find optimal batch size
   python scripts/find_optimal_batch_size.py
   ```

3. **Use model caching**:
   ```bash
   # Pre-download models
   python -c "from models.base_model import AuthenticityClassifier; AuthenticityClassifier()"
   ```

## 🔒 Security Considerations

- API rate limiting implemented
- File size limits enforced
- Input validation on all endpoints
- No sensitive data in logs
- Secure model checkpoints

## 📋 Production Deployment

### Docker Production

```bash
# Production docker-compose
docker-compose -f docker-compose.prod.yml up -d

# With environment override
docker-compose --env-file .env.prod up -d
```

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    spec:
      containers:
      - name: ml-service
        image: human-art-guardians-ml:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Load Balancing

For high-traffic deployments:

```bash
# Scale with Docker Compose
docker-compose up --scale ml-service=3

# Use nginx for load balancing
# See nginx.conf.example
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: See the `/docs` folder
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@humanartguardians.com

## 🗺️ Roadmap

- [ ] Multi-modal analysis (text + image)
- [ ] Advanced adversarial detection
- [ ] Real-time streaming analysis
- [ ] Mobile app integration
- [ ] Blockchain verification
- [ ] Custom model architectures
- [ ] Federated learning support

---

**Built with ❤️ for the Human Art Guardians community** 