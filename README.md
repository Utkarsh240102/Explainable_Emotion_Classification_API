# 🎭 Explainable Emotion Classification API

A production-ready, AI-powered emotion classification system that analyzes text to detect emotions and provides detailed, explainable predictions. Built with RoBERTa (GoEmotions model), FastAPI, and React.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Emotion Types](#emotion-types)
- [Frontend Interface](#frontend-interface)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Examples](#examples)
- [License](#license)

---

## 🎯 Overview

This system uses state-of-the-art Natural Language Processing (NLP) to classify text into one of 28 emotion categories from the GoEmotions dataset. Unlike traditional black-box models, this API provides:

- **Explainable Predictions**: Detailed explanations of why a particular emotion was detected
- **Clause-Level Analysis**: Analyzes complex sentences at the clause level to detect mixed or opposing emotions
- **Confidence Scores**: Probability distributions across all emotion categories
- **Real-time Processing**: Fast inference with sub-second response times
- **Production-Ready**: Complete with health checks, logging, error handling, and CORS support

### Use Cases

- **Customer Feedback Analysis**: Understand customer emotions in reviews and support tickets
- **Social Media Monitoring**: Analyze sentiment and emotions in social media posts
- **Content Moderation**: Detect negative emotions in user-generated content
- **Mental Health Applications**: Analyze emotional patterns in text
- **Chatbot Enhancement**: Improve chatbot responses by understanding user emotions
- **Market Research**: Analyze emotional responses to products and services

---

## ✨ Key Features

### 🧠 Advanced Emotion Detection

- **28 Emotion Categories**: Admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
- **Multi-Emotion Detection**: Identifies mixed, opposing, and ambiguous emotions
- **Clause-Level Analysis**: Breaks down complex sentences to detect emotion shifts

### 🔍 Explainability

- **Keyword Matching**: Identifies emotion-specific keywords in the text
- **Rule-Based Explanations**: Uses linguistic rules to explain predictions
- **Negation Detection**: Handles negated emotions (e.g., "not happy")
- **Context Analysis**: Considers sentence structure and conjunctions

### 🔧 Reasoning Assistant

- **Expectation vs Experience Detection**: Distinguishes between anticipated and felt emotions
- **Emotional Exhaustion Mapping**: Maps fatigue to appropriate emotions
- **Contrast Detection**: Identifies contrasting clauses using "but", "however", etc.
- **Confidence Calibration**: Labels ambiguous cases when confidence is low (<60%)

### 🚀 Production Features

- **FastAPI Backend**: Modern, fast (high-performance) web framework
- **React Frontend**: Interactive, responsive UI with real-time predictions
- **Docker Support**: Containerized deployment with single command
- **Health Monitoring**: `/health` endpoint for service monitoring
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **CORS Enabled**: Ready for cross-origin requests

---

## 🛠️ Technology Stack

### Backend

- **Python 3.13+**
- **FastAPI 0.115+**: Modern web framework for building APIs
- **PyTorch 2.10+**: Deep learning framework
- **Transformers 4.47+**: Hugging Face library for NLP models
- **Pydantic 2.10+**: Data validation using Python type annotations
- **Uvicorn**: ASGI server for FastAPI

### Frontend

- **React 18.3+**: UI library
- **Vite 5.3+**: Build tool and dev server
- **Axios 1.7+**: HTTP client for API requests
- **Recharts 2.12+**: Charting library for visualizations

### Model

- **RoBERTa-base**: Transformer architecture
- **GoEmotions Dataset**: Trained on Reddit comments (58k examples)
- **SamLowe/roberta-base-go_emotions**: Pre-trained model from Hugging Face

---

## 🏗️ Architecture

### System Workflow

```
User Input → Validation → Preprocessing → Tokenization → BERT Inference
→ Probability Calculation → Clause Analysis → Reasoning Fixes → Explanation → Response
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ EmotionForm  │  │ EmotionResults│  │ Visualizations │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                        │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   main.py  │  │  schemas.py  │  │    tokenizer.py   │  │
│  └────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  model.py  │  │  explain.py  │  │ clause_analyzer.py│  │
│  └────────────┘  └──────────────┘  └────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          reasoning_assistant.py                        │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              RoBERTa Model (GoEmotions)                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Pretrained transformer with 28-emotion classifier     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| **main.py** | FastAPI app entry point, endpoint definitions, request orchestration |
| **model.py** | RoBERTa model loading, inference, logits-to-probability conversion |
| **tokenizer.py** | Text preprocessing, tokenization, padding, attention masks |
| **explain.py** | Rule-based explanation generation, keyword matching |
| **clause_analyzer.py** | Clause splitting, mixed emotion detection, emotion shift analysis |
| **reasoning_assistant.py** | Post-processing fixes, logical error correction, confidence calibration |
| **schemas.py** | Pydantic models for request/response validation |

---

## 📁 Project Structure

```
Explainable_Emotion_Classification_API/
│
├── backend/                          # Backend API
│   ├── __init__.py                  # Package initialization
│   ├── main.py                      # FastAPI application entry point
│   ├── model.py                     # RoBERTa emotion classifier
│   ├── tokenizer.py                 # Text tokenization
│   ├── explain.py                   # Explanation generator
│   ├── clause_analyzer.py           # Clause-level emotion analysis
│   ├── reasoning_assistant.py       # Post-processing reasoning fixes
│   └── schemas.py                   # Pydantic request/response models
│
├── frontend/                         # React frontend
│   ├── src/
│   │   ├── App.jsx                  # Main application component
│   │   ├── main.jsx                 # React entry point
│   │   ├── components/
│   │   │   ├── EmotionForm.jsx      # Input form component
│   │   │   ├── EmotionResults.jsx   # Results display component
│   │   │   ├── EmotionForm.css      # Form styles
│   │   │   └── EmotionResults.css   # Results styles
│   │   └── services/
│   │       └── api.js               # API client
│   ├── index.html                   # HTML entry point
│   ├── package.json                 # Node.js dependencies
│   └── vite.config.js               # Vite configuration
│
├── myenv/                            # Python virtual environment
│
├── requirements.txt                  # Python dependencies
├── dockerfile                        # Docker container definition
└── README.md                         # Project documentation (this file)
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.13+** (or 3.10+)
- **Node.js 18+** and npm (for frontend)
- **Git**
- **Docker** (optional, for containerized deployment)

### Option 1: Local Installation

#### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Explainable_Emotion_Classification_API
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv myenv
   myenv\Scripts\activate

   # Linux/Mac
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server**:
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at:
   - API: http://localhost:8000
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```

   The frontend will be available at http://localhost:5173

### Option 2: Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t emotion-classification-api .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 emotion-classification-api
   ```

   The API will be available at http://localhost:8000

---

## 💻 Usage

### Using the Web Interface

1. Open your browser and navigate to http://localhost:5173 (frontend)
2. Enter text in the input field
3. Click "Analyze Emotion"
4. View the results:
   - Primary emotion and confidence score
   - Emotion type (single, mixed, opposing, ambiguous)
   - Detailed explanation
   - Probability distribution chart
   - Clause-level breakdown (for complex sentences)

### Using the API Directly

#### Python Example

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "text": "I was so excited about the concert, but it got cancelled last minute."
}

response = requests.post(url, json=data)
result = response.json()

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']}")
print(f"Explanation: {result['explanation']}")
print(f"All emotions: {result['all_emotions']}")
```

#### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this amazing project!"}'
```

#### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'I am so happy and grateful for this opportunity!'
  })
});

const result = await response.json();
console.log(result);
```

---

## 🔌 API Endpoints

### `POST /predict`

Predict emotion from input text with detailed explanation.

**Request Body**:
```json
{
  "text": "Your text here"
}
```

**Response**:
```json
{
  "emotion": "joy",
  "confidence": 0.92,
  "all_emotions": {
    "joy": 0.92,
    "excitement": 0.04,
    "gratitude": 0.02,
    "neutral": 0.01,
    "admiration": 0.01
  },
  "explanation": "The text contains strong positive indicators like 'happy' and 'grateful', suggesting a joyful emotion.",
  "emotion_type": "single",
  "primary_emotions": ["joy"],
  "clauses": null
}
```

**For Complex Text with Multiple Emotions**:
```json
{
  "emotion": "disappointment",
  "confidence": 0.68,
  "all_emotions": { ... },
  "explanation": "The text shows opposing emotions: initial excitement followed by disappointment...",
  "emotion_type": "opposing",
  "primary_emotions": ["excitement", "disappointment"],
  "clauses": [
    {
      "text": "I was so excited about the concert",
      "emotion": "excitement",
      "confidence": 0.89,
      "all_probabilities": { ... }
    },
    {
      "text": "but it got cancelled last minute",
      "emotion": "disappointment",
      "confidence": 0.76,
      "all_probabilities": { ... }
    }
  ]
}
```

### `GET /health`

Health check endpoint for monitoring.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "All components loaded successfully"
}
```

### `GET /emotions`

Get list of all supported emotion labels.

**Response**:
```json
{
  "emotions": [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
  ],
  "count": 28
}
```

### `GET /`

Root endpoint with API information.

**Response**:
```json
{
  "message": "Explainable Emotion Classification API",
  "version": "1.0.0",
  "description": "BERT-based emotion classification with explainable predictions",
  "endpoints": {
    "POST /predict": "Classify emotion in text",
    "GET /health": "Health check",
    "GET /emotions": "List supported emotions",
    "GET /docs": "API documentation (Swagger UI)",
    "GET /redoc": "API documentation (ReDoc)"
  }
}
```

---

## 🤖 Model Details

### Base Model

- **Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Dataset**: GoEmotions (58,000 Reddit comments)
- **Hugging Face**: `SamLowe/roberta-base-go_emotions`
- **Architecture**: Transformer-based sequence classification
- **Parameters**: ~125M parameters
- **Input**: Text (max 128 tokens)
- **Output**: 28-class probability distribution

### Model Pipeline

1. **Tokenization**: Text → Token IDs + Attention Mask
2. **Embedding**: Token IDs → Embeddings (768-dimensional)
3. **Transformer Layers**: 12 layers of self-attention
4. **Classification Head**: Embeddings → 28 logits
5. **Softmax**: Logits → Probabilities
6. **Prediction**: argmax(probabilities) → Emotion

### Performance

- **Inference Time**: ~100-300ms per prediction (CPU)
- **Inference Time**: ~30-50ms per prediction (GPU)
- **Model Size**: ~500MB
- **Max Input Length**: 512 characters (recommended), 128 tokens (model limit)

---

## 🎨 Emotion Types

The system classifies predictions into four emotion types:

### 1. **Single Emotion**
- One dominant emotion with high confidence
- Example: "I'm so happy!" → `joy` (95%)

### 2. **Mixed Emotion**
- Multiple related emotions present
- Example: "I'm excited but nervous" → `excitement` + `nervousness`

### 3. **Opposing Emotion**
- Contrasting emotions (separated by "but", "however", etc.)
- Example: "I was excited, but now I'm disappointed"
- **Priority**: Post-contrast emotion dominates

### 4. **Ambiguous Emotion**
- Low confidence (<60%) across all emotions
- Unable to determine clear emotion
- Example: "It is what it is"

---

## 🎨 Frontend Interface

### Features

- **Clean, Modern UI**: Responsive design with gradient backgrounds
- **Real-time Analysis**: Instant emotion predictions
- **Visual Feedback**: 
  - Emotion emoji indicators
  - Confidence percentage with color coding
  - Bar chart showing all emotion probabilities
- **Clause Breakdown**: For complex sentences, shows emotion analysis for each clause
- **Explanation Section**: Detailed human-readable explanation
- **Error Handling**: User-friendly error messages
- **Loading States**: Spinner and status indicators

### Technology

- **React 18**: Component-based UI
- **Vite**: Fast build tool and dev server
- **CSS3**: Custom styling with animations
- **Recharts**: Interactive charts for probability visualization
- **Axios**: HTTP client for API communication

---

## 🐳 Docker Deployment

### Dockerfile

The project includes a production-ready Dockerfile:

```dockerfile
FROM python:3.13.5

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t emotion-api .

# Run container
docker run -d -p 8000:8000 --name emotion-api emotion-api

# View logs
docker logs -f emotion-api

# Stop container
docker stop emotion-api

# Remove container
docker rm emotion-api
```

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## 🔧 Development

### Running Tests

```bash
# Backend tests (if implemented)
pytest backend/tests/

# Frontend tests (if implemented)
cd frontend
npm test
```

### Code Quality

```bash
# Python linting
flake8 backend/
black backend/

# JavaScript linting
cd frontend
npm run lint
```

### Adding New Emotions

1. Update `EMOTION_KEYWORDS` in [backend/explain.py](backend/explain.py)
2. Add explanation patterns in `ExplanationGenerator`
3. Update frontend emotion colors/icons if needed

### Modifying Reasoning Rules

Edit [backend/reasoning_assistant.py](backend/reasoning_assistant.py):
- Add new rule detection methods
- Update `apply_reasoning_fixes()` logic
- Test with edge cases

---

## 📊 Examples

### Example 1: Simple Positive Emotion

**Input**: 
```
"I'm so happy and grateful for this amazing opportunity!"
```

**Output**:
```json
{
  "emotion": "joy",
  "confidence": 0.94,
  "emotion_type": "single",
  "explanation": "The text contains strong positive words like 'happy', 'grateful', and 'amazing', clearly indicating joy."
}
```

### Example 2: Opposing Emotions

**Input**: 
```
"I thought this job would make me happy, but I'm actually exhausted and disappointed."
```

**Output**:
```json
{
  "emotion": "disappointment",
  "confidence": 0.72,
  "emotion_type": "opposing",
  "primary_emotions": ["optimism", "disappointment"],
  "explanation": "The text shows opposing emotions: anticipated happiness (optimism) vs. experienced disappointment. The latter dominates."
}
```

### Example 3: Mixed Emotions

**Input**: 
```
"I'm excited about the move but also sad to leave my friends."
```

**Output**:
```json
{
  "emotion": "excitement",
  "confidence": 0.68,
  "emotion_type": "mixed",
  "primary_emotions": ["excitement", "sadness"],
  "clauses": [
    {"text": "I'm excited about the move", "emotion": "excitement", "confidence": 0.91},
    {"text": "but also sad to leave my friends", "emotion": "sadness", "confidence": 0.88}
  ]
}
```

### Example 4: Neutral/Ambiguous

**Input**: 
```
"It is what it is."
```

**Output**:
```json
{
  "emotion": "neutral",
  "confidence": 0.82,
  "emotion_type": "single",
  "explanation": "The text is neutral with no strong emotional indicators."
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint rules for JavaScript
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

---

## 📝 License

This project is provided as-is for educational and commercial use. Please ensure compliance with:
- **Hugging Face Model License**: Check the license for `SamLowe/roberta-base-go_emotions`
- **GoEmotions Dataset License**: Apache 2.0
- **Dependencies**: Check individual package licenses

---

## 🙏 Acknowledgments

- **Hugging Face**: For the Transformers library and model hosting
- **Google Research**: For the GoEmotions dataset
- **SamLowe**: For the fine-tuned RoBERTa model
- **FastAPI**: For the excellent web framework
- **React Team**: For the frontend framework

---

## 📧 Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers

---

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Batch prediction endpoint
- [ ] Emotion timeline visualization
- [ ] User feedback collection
- [ ] Model fine-tuning interface
- [ ] REST API rate limiting
- [ ] WebSocket support for streaming
- [ ] Export results to CSV/JSON
- [ ] Integration with popular NLP libraries
- [ ] Mobile app (React Native)

---

**Built with ❤️ using RoBERTa, FastAPI, and React**
