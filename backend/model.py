"""
BERT Model module for emotion classification.

Handles:
1. Loading pretrained BERT model for sequence classification
2. Model inference (forward pass)
3. Logits to probability conversion
4. Prediction selection
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """
    DistilBERT-based emotion classifier.
    
    Predicts one of seven emotions:
    - Anger
    - Disgust
    - Fear
    - Joy
    - Neutral
    - Sadness
    - Surprise
    """
    
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    def __init__(
        self,
        model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion",
        device: str = None
    ):
        """
        Initialize the emotion classifier.
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on ('cpu' or 'cuda')
        """
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading DistilBERT model on device: {self.device}")
        
        # Load DistilBERT model for sequence classification
        # Don't specify num_labels - use model's trained configuration
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode (disables dropout)
        self.model.eval()
        
        # Get actual number of labels from model
        self.num_labels = self.model.config.num_labels
        
        # Update emotion labels to match model's actual labels
        if self.num_labels == 6:
            # This model has 6 emotions (no 'neutral')
            self.EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        logger.info(f"Model loaded successfully with {self.num_labels} emotion classes: {self.EMOTION_LABELS}")
    
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward pass through BERT to get raw logits.
        
        Args:
            input_ids: Token IDs from tokenizer (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            
        Returns:
            Raw logits for each emotion class (batch_size, num_labels)
        """
        # Move tensors to model device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Disable gradient computation (inference mode)
        with torch.no_grad():
            # Forward pass through BERT
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract logits from model output
            logits = outputs.logits
        
        return logits
    
    def logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert raw logits to probability scores using softmax.
        
        Args:
            logits: Raw logits from model (batch_size, num_labels)
            
        Returns:
            Probability scores (batch_size, num_labels)
        """
        # Apply softmax along the class dimension (dim=1)
        probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, any]:
        """
        Complete prediction pipeline.
        
        Pipeline:
        1. Get logits from BERT
        2. Convert logits to probabilities
        3. Select emotion with highest probability
        4. Extract confidence score
        5. Return all probabilities for explanation
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Dictionary containing:
                - emotion: Predicted emotion label (str)
                - confidence: Confidence score (float)
                - all_probabilities: Dict mapping emotion -> probability
                - logits: Raw logits (for debugging)
        """
        # Step 1: Get raw logits
        logits = self.get_logits(input_ids, attention_mask)
        
        # Step 2: Convert to probabilities
        probabilities = self.logits_to_probabilities(logits)
        
        # Move to CPU and convert to numpy for easier handling
        probabilities_cpu = probabilities.cpu().squeeze().numpy()
        logits_cpu = logits.cpu().squeeze().numpy()
        
        # Step 3: Find emotion with highest probability
        predicted_idx = probabilities_cpu.argmax()
        predicted_emotion = self.EMOTION_LABELS[predicted_idx]
        
        # Step 4: Extract confidence score
        confidence = float(probabilities_cpu[predicted_idx])
        
        # Step 5: Create probability distribution for all emotions
        all_probabilities = {
            emotion: float(prob)
            for emotion, prob in zip(self.EMOTION_LABELS, probabilities_cpu)
        }
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'logits': logits_cpu.tolist()
        }
    
    def get_emotion_labels(self) -> list:
        """Return list of all emotion labels."""
        return self.EMOTION_LABELS.copy()
