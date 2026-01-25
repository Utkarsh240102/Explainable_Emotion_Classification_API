"""
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field, validator


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


class EmotionPrediction(BaseModel):
    """Response schema for emotion classification."""
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    explanation: str = Field(..., description="Human-readable explanation of the prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emotion": "joy",
                "confidence": 0.92,
                "explanation": "The text contains positive words like 'happy' and 'excited', indicating a joyful emotion."
            }
        }


class HealthCheck(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool
    message: str
