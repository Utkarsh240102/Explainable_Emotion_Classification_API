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
from transformers import RobertaForSequenceClassification
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """
    RoBERTa-based emotion classifier using GoEmotions.
    
    Predicts one or multiple emotions from 28 categories:
    Includes emotions like joy, sadness, anger, fear, surprise, love,
    gratitude, admiration, confusion, and many more nuanced emotions.
    """
    
    # GoEmotions has 28 emotion labels (27 emotions + neutral)
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
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
        
        logger.info(f"Loading RoBERTa GoEmotions model on device: {self.device}")
        
        # Load RoBERTa model for sequence classification (GoEmotions)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode (disables dropout)
        self.model.eval()
        
        # Get actual number of labels from model
        self.num_labels = self.model.config.num_labels
        
        logger.info(f"Model loaded successfully with {self.num_labels} emotion classes")
    
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
            # Forward pass through RoBERTa
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
    ) -> Dict[str, Any]:
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
    #function to get emotion labels
    def get_emotion_labels(self) -> list:
        """Return list of all emotion labels."""
        return self.EMOTION_LABELS.copy()
