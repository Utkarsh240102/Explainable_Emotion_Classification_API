"""
FastAPI Application Entry Point.

This orchestrates the complete workflow:
User Input → Validation → Preprocessing → Tokenization → BERT Inference
→ Probability Calculation → Prediction → Explanation → Response
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from backend.schemas import TextInput, EmotionPrediction, HealthCheck
from backend.tokenizer import EmotionTokenizer
from backend.model import EmotionClassifier
from backend.explain import ExplanationGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model components
tokenizer = None
model = None
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads models on startup and cleans up on shutdown.
    """
    # Startup: Load models
    global tokenizer, model, explainer
    
    logger.info("=" * 60)
    logger.info("Starting Explainable Emotion Classification API")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading emotion-specific tokenizer...")
        tokenizer = EmotionTokenizer(
            model_name="bhadresh-savani/distilbert-base-uncased-emotion",
            max_length=128
        )
        logger.info(f"✓ Tokenizer loaded successfully")
        
        logger.info("Loading pre-trained emotion model...")
        model = EmotionClassifier(
            model_name="bhadresh-savani/distilbert-base-uncased-emotion",
            device=None  # Auto-detect device
        )
        logger.info(f"✓ Model loaded successfully")
        
        logger.info("Initializing explanation generator...")
        explainer = ExplanationGenerator()
        logger.info(f"✓ Explainer initialized successfully")
        
        logger.info("=" * 60)
        logger.info("API is ready to accept requests!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down API...")
    logger.info("Cleanup completed.")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Explainable Emotion Classification API",
    description=(
        "A production-ready API for emotion classification using BERT. "
        "Classifies text into one of five emotions (anger, joy, sadness, fear, neutral) "
        "and provides explainable predictions with confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
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


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model loading status
    """
    model_loaded = all([tokenizer is not None, model is not None, explainer is not None])
    
    return HealthCheck(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        message="All components loaded successfully" if model_loaded else "Models not loaded"
    )


@app.get("/emotions", tags=["Information"])
async def get_emotions():
    """
    Get list of supported emotion labels.
    
    Returns:
        List of emotion labels that the model can predict
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet"
        )
    
    return {
        "emotions": model.get_emotion_labels(),
        "count": len(model.get_emotion_labels())
    }


@app.post("/predict", response_model=EmotionPrediction, tags=["Prediction"])
async def predict_emotion(input_data: TextInput):
    """
    Predict emotion from input text with explanation.
    
    Complete workflow:
    1. Receive and validate input text
    2. Preprocess and tokenize text
    3. Run BERT inference to get logits
    4. Convert logits to probabilities
    5. Select emotion with highest probability
    6. Generate human-readable explanation
    7. Return structured response
    
    Args:
        input_data: TextInput object containing the text to classify
        
    Returns:
        EmotionPrediction object with emotion, confidence, and explanation
        
    Raises:
        HTTPException: If models are not loaded or prediction fails
    """
    # Check if models are loaded
    if any(component is None for component in [tokenizer, model, explainer]):
        logger.error("Prediction request received but models not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading. Please try again in a moment."
        )
    
    try:
        # Step 1: Extract text from validated input
        text = input_data.text
        logger.info(f"Received prediction request for text: '{text[:50]}...'")
        
        # Step 2: Tokenize text
        logger.debug("Tokenizing input text...")
        tokenized = tokenizer.tokenize(text)
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        tokens = tokenized['tokens']
        
        logger.debug(f"Tokenization complete. Tokens: {tokens[:10]}...")
        
        # Step 3: Run model inference
        logger.debug("Running BERT inference...")
        prediction = model.predict(input_ids, attention_mask)
        
        emotion = prediction['emotion']
        confidence = prediction['confidence']
        all_probabilities = prediction['all_probabilities']
        
        logger.info(f"Prediction: {emotion} (confidence: {confidence:.4f})")
        logger.debug(f"All probabilities: {all_probabilities}")
        
        # Step 4: Generate explanation
        logger.debug("Generating explanation...")
        explanation = explainer.generate_explanation(
            text=text,
            emotion=emotion,
            confidence=confidence,
            all_probabilities=all_probabilities
        )
        
        logger.info(f"Explanation: {explanation[:100]}...")
        
        # Step 5: Create response
        response = EmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            explanation=explanation
        )
        
        logger.info("Prediction completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred",
            "error": str(exc)
        }
    )


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
