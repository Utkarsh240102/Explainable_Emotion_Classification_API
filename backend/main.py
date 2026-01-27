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

from backend.schemas import TextInput, EmotionPrediction, HealthCheck, ClauseEmotion
from backend.tokenizer import EmotionTokenizer
from backend.model import EmotionClassifier
from backend.explain import ExplanationGenerator
from backend.clause_analyzer import ClauseAnalyzer
from backend.reasoning_assistant import EmotionReasoningAssistant

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
clause_analyzer = None
reasoning_assistant = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads models on startup and cleans up on shutdown.
    """
    # Startup: Load models
    global tokenizer, model, explainer, clause_analyzer, reasoning_assistant
    
    logger.info("=" * 60)
    logger.info("Starting Explainable Emotion Classification API")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading RoBERTa tokenizer for GoEmotions...")
        tokenizer = EmotionTokenizer(
            model_name="SamLowe/roberta-base-go_emotions",
            max_length=128
        )
        logger.info(f"✓ Tokenizer loaded successfully")
        
        logger.info("Loading RoBERTa GoEmotions model...")
        model = EmotionClassifier(
            model_name="SamLowe/roberta-base-go_emotions",
            device=None  # Auto-detect device
        )
        logger.info(f"✓ Model loaded successfully")
        
        logger.info("Initializing explanation generator...")
        explainer = ExplanationGenerator()
        logger.info(f"✓ Explainer initialized successfully")
        
        logger.info("Initializing clause analyzer...")
        clause_analyzer = ClauseAnalyzer()
        logger.info(f"✓ Clause analyzer initialized successfully")
        
        logger.info("Initializing reasoning assistant...")
        reasoning_assistant = EmotionReasoningAssistant()
        logger.info(f"✓ Reasoning assistant initialized successfully")
        
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
    model_loaded = all([
        tokenizer is not None,
        model is not None,
        explainer is not None,
        clause_analyzer is not None
    ])
    
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
    if any(component is None for component in [tokenizer, model, explainer, clause_analyzer]):
        logger.error("Prediction request received but models not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are still loading. Please try again in a moment."
        )
    
    try:
        # Step 1: Extract text from validated input
        text = input_data.text
        logger.info(f"Received prediction request for text: '{text[:50]}...'")
        
        # Step 2: Check if clause-level analysis is needed
        needs_clause_analysis = clause_analyzer.should_analyze_clauses(text)
        logger.debug(f"Clause analysis needed: {needs_clause_analysis}")
        
        if needs_clause_analysis:
            # === CLAUSE-LEVEL ANALYSIS PATH ===
            logger.info("Performing clause-level emotion analysis")
            
            # Split into clauses
            clauses = clause_analyzer.split_into_clauses(text)
            logger.debug(f"Split into {len(clauses)} clauses: {clauses}")
            
            # Analyze each clause
            clause_emotions = []
            for clause_text in clauses:
                # Tokenize clause
                tokenized = tokenizer.tokenize(clause_text)
                
                # Predict emotion for clause
                prediction = model.predict(tokenized['input_ids'], tokenized['attention_mask'])
                
                clause_emotions.append({
                    'text': clause_text,
                    'emotion': prediction['emotion'],
                    'confidence': prediction['confidence'],
                    'all_probabilities': prediction['all_probabilities']
                })
                
                logger.debug(f"Clause: '{clause_text[:30]}...' → {prediction['emotion']} ({prediction['confidence']:.2f})")
            
            # Detect emotion shifts
            shift_analysis = clause_analyzer.detect_emotion_shift(clause_emotions)
            
            # Get primary emotion with emotion_type
            primary_emotion, avg_confidence, emotion_type = clause_analyzer.get_primary_emotion(clause_emotions)
            
            # Get all probabilities (averaged across clauses)
            all_emotions_avg = {}
            for emotion_label in model.get_emotion_labels():
                probs = [c['all_probabilities'].get(emotion_label, 0) for c in clause_emotions]
                all_emotions_avg[emotion_label] = sum(probs) / len(probs)
            
            # Generate explanation for mixed emotions
            explanation = explainer.generate_clause_level_explanation(
                text=text,
                clause_emotions=clause_emotions,
                shift_analysis=shift_analysis,
                primary_emotion=primary_emotion,
                emotion_type=emotion_type
            )
            
            # Step 6: Apply reasoning fixes
            logger.debug("Applying reasoning fixes...")
            raw_output = {
                'emotion': primary_emotion,
                'confidence': avg_confidence,
                'all_emotions': all_emotions_avg,
                'explanation': explanation,
                'emotion_type': emotion_type,
                'clauses': [{
                    'text': c['text'],
                    'emotion': c['emotion'],
                    'confidence': c['confidence']
                } for c in clause_emotions]
            }
            
            fixed_output = reasoning_assistant.apply_reasoning_fixes(text, raw_output)
            
            # Create clause emotion objects from fixed output
            clause_objects = [
                ClauseEmotion(
                    text=c['text'],
                    emotion=c['emotion'],
                    confidence=c['confidence']
                )
                for c in fixed_output['clauses']
            ]
            
            # Create response with fixed reasoning
            response = EmotionPrediction(
                emotion=fixed_output['primary_emotion'],
                confidence=fixed_output['confidence'],
                all_emotions=all_emotions_avg,
                explanation=fixed_output['explanation'],
                emotion_type=fixed_output['emotion_type'],
                clauses=clause_objects
            )
            
            logger.info(f"Clause-level analysis complete: {fixed_output['emotion_type']} emotions")
            return response
            
        else:
            # === STANDARD SINGLE-CLAUSE ANALYSIS PATH ===
            logger.debug("Performing standard single-clause analysis")
            
            # Step 3: Tokenize text
            tokenized = tokenizer.tokenize(text)
            
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            tokens = tokenized['tokens']
            
            logger.debug(f"Tokenization complete. Tokens: {tokens[:10]}...")
            
            # Step 4: Run model inference
            logger.debug("Running model inference...")
            prediction = model.predict(input_ids, attention_mask)
            
            emotion = prediction['emotion']
            confidence = prediction['confidence']
            all_probabilities = prediction['all_probabilities']
            
            logger.info(f"Prediction: {emotion} (confidence: {confidence:.4f})")
            logger.debug(f"All probabilities: {all_probabilities}")
            
            # Step 5: Generate explanation
            logger.debug("Generating explanation...")
            explanation = explainer.generate_explanation(
                text=text,
                emotion=emotion,
                confidence=confidence,
                all_probabilities=all_probabilities
            )
            
            logger.info(f"Explanation: {explanation[:100]}...")
            
            # Step 6: Apply reasoning fixes
            logger.debug("Applying reasoning fixes...")
            raw_output = {
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': all_probabilities,
                'explanation': explanation,
                'emotion_type': 'single',
                'clauses': []
            }
            
            fixed_output = reasoning_assistant.apply_reasoning_fixes(text, raw_output)
            
            # Create response with fixed reasoning
            response = EmotionPrediction(
                emotion=fixed_output['primary_emotion'],
                confidence=fixed_output['confidence'],
                all_emotions=all_probabilities,
                explanation=fixed_output['explanation'],
                emotion_type=fixed_output['emotion_type'],
                clauses=None if not fixed_output['clauses'] else [
                    ClauseEmotion(
                        text=c['text'],
                        emotion=c['emotion'],
                        confidence=c['confidence']
                    )
                    for c in fixed_output['clauses']
                ]
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
