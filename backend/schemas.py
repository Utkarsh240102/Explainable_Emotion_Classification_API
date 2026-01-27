"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional


class TextInput(BaseModel):
    """Request schema for emotion classification."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Input text for emotion classification"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate that text is not empty after stripping whitespace."""
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class ClauseEmotion(BaseModel):
    """Emotion prediction for a single clause."""
    text: str = Field(..., description="The clause text")
    emotion: str = Field(..., description="Predicted emotion for this clause")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class EmotionPrediction(BaseModel):
    """Response schema for emotion classification."""
    emotion: str = Field(..., description="Predicted emotion label (primary emotion if mixed)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    all_emotions: dict = Field(..., description="Probability scores for all emotions")
    explanation: str = Field(..., description="Human-readable explanation of the prediction")
    
    # Clause-level analysis (optional, only for complex text)
    emotion_type: Optional[str] = Field(None, description="Type: 'single', 'mixed', or 'opposing'")
    clauses: Optional[List[ClauseEmotion]] = Field(None, description="Clause-level emotion breakdown")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emotion": "joy",
                "confidence": 0.92,
                "all_emotions": {
                    "sadness": 0.02,
                    "joy": 0.92,
                    "love": 0.03,
                    "anger": 0.01,
                    "fear": 0.01,
                    "surprise": 0.01
                },
                "explanation": "The text contains positive words like 'happy' and 'excited', indicating a joyful emotion.",
                "emotion_type": "single",
                "clauses": None
            }
        }


class HealthCheck(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool
    message: str
