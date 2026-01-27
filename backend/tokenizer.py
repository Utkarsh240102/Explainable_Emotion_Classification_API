"""
BERT Tokenizer module for text preprocessing.

Handles:
1. Text preprocessing (cleaning and normalization)
2. Tokenization using BERT tokenizer
3. Generation of input_ids and attention_mask
4. Special token handling ([CLS] and [SEP])
"""

from transformers import RobertaTokenizer
from typing import Dict
import torch


class EmotionTokenizer:
    """
    Handles tokenization for emotion classification using BERT tokenizer.
    """
    
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions", max_length: int = 128):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: Name of the pretrained RoBERTa model/tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        text = text.strip()
        return text
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Convert text to token IDs with attention mask.
        
        Process:
        1. Preprocess the text
        2. Tokenize using BERT tokenizer
        3. Add special tokens [CLS] at start and [SEP] at end
        4. Generate attention mask (1 for real tokens, 0 for padding)
        5. Pad/truncate to max_length
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs (tensor)
                - attention_mask: Attention mask (tensor)
                - tokens: List of tokens (for debugging/explanation)
        """
        # Step 1: Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Step 2-5: Tokenize with BERT tokenizer
        encoding = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,          # Add [CLS] and [SEP]
            max_length=self.max_length,       # Maximum sequence length
            padding='max_length',             # Pad to max_length
            truncation=True,                  # Truncate if longer than max_length
            return_attention_mask=True,       # Return attention mask
            return_tensors='pt'               # Return PyTorch tensors
        )
        
        # Get the actual tokens (for explanation purposes)
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding['input_ids'][0].tolist()
        )
        
        return {
            'input_ids': encoding['input_ids'],          # Shape: (1, max_length)
            'attention_mask': encoding['attention_mask'], # Shape: (1, max_length)
            'tokens': tokens                              # List of token strings
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size
