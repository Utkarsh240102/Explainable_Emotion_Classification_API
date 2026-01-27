"""
Emotion Reasoning Assistant Module.

Post-processes raw emotion classification outputs to fix logical errors
using linguistic rules. Does NOT re-run classification, only fixes reasoning.
"""

from typing import Dict, List, Any, Optional
import re


class EmotionReasoningAssistant:
    """
    Fixes reasoning errors in emotion classification outputs.
    
    Rules:
    1. Expectation vs Experience - Detect anticipated emotions
    2. Emotional Exhaustion - Map tired→disappointment, not neutral
    3. Clause Deduplication - Remove duplicate clauses
    4. Final Emotion Selection - Experienced > Anticipatory
    5. Honesty Over Certainty - Label ambiguous when confidence < 0.6
    """
    
    # Expectation/belief indicators
    ANTICIPATORY_WORDS = [
        'thought', 'expected', 'hoped', 'assumed', 'believed',
        'would make me', 'should make me', 'will make me',
        'expecting', 'hoping', 'assuming', 'thinking',
        'supposed to', 'meant to'
    ]
    
    # Emotional exhaustion words
    EXHAUSTION_WORDS = [
        'tired', 'drained', 'exhausted', 'burned out', 'burnt out',
        'worn out', 'weary', 'fatigued', 'depleted', 'spent'
    ]
    
    # Contrast words (for dominance rule)
    CONTRAST_WORDS = [
        'but', 'yet', 'however', 'though', 'although',
        'even though', 'despite', 'nevertheless', 'nonetheless'
    ]
    
    def __init__(self):
        """Initialize the reasoning assistant."""
        pass
    
    def is_anticipatory(self, text: str) -> bool:
        """
        RULE 1: Detect if text expresses anticipated emotion.
        
        Args:
            text: Clause text
            
        Returns:
            True if anticipatory, False if experienced
        """
        text_lower = text.lower()
        return any(word in text_lower for word in self.ANTICIPATORY_WORDS)
    
    def contains_exhaustion(self, text: str) -> bool:
        """
        RULE 2: Detect emotional exhaustion words.
        
        Args:
            text: Clause text
            
        Returns:
            True if contains exhaustion indicators
        """
        text_lower = text.lower()
        return any(word in text_lower for word in self.EXHAUSTION_WORDS)
    
    def detect_contrast_position(self, text: str, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        RULE 6 (NEW): Detect contrast words and classify clauses as pre/post contrast.
        
        Args:
            text: Full original text
            clauses: List of clause dictionaries
            
        Returns:
            Dict with contrast_detected, pre_contrast_indices, post_contrast_indices
        """
        text_lower = text.lower()
        
        # Check if any contrast word exists
        has_contrast = any(word in text_lower for word in self.CONTRAST_WORDS)
        
        if not has_contrast or len(clauses) < 2:
            return {
                'contrast_detected': False,
                'pre_contrast_indices': [],
                'post_contrast_indices': list(range(len(clauses)))
            }
        
        # Find the position of the first contrast word
        contrast_position = len(text)
        for word in self.CONTRAST_WORDS:
            pos = text_lower.find(word)
            if pos != -1 and pos < contrast_position:
                contrast_position = pos
        
        # Classify each clause as pre or post contrast
        pre_indices = []
        post_indices = []
        
        for i, clause in enumerate(clauses):
            clause_text = clause.get('text', '')
            # Find where this clause appears in the original text
            clause_pos = text_lower.find(clause_text.lower())
            
            if clause_pos < contrast_position:
                pre_indices.append(i)
            else:
                post_indices.append(i)
        
        return {
            'contrast_detected': True,
            'pre_contrast_indices': pre_indices,
            'post_contrast_indices': post_indices
        }
    
    def apply_contrast_weighting(
        self,
        clauses: List[Dict[str, Any]],
        contrast_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        RULE 6 (NEW): Apply contrast dominance weighting.
        
        Post-contrast emotions get boosted, pre-contrast get reduced.
        
        Args:
            clauses: List of clause dictionaries
            contrast_info: Output from detect_contrast_position
            
        Returns:
            Clauses with adjusted confidence scores
        """
        if not contrast_info['contrast_detected']:
            return clauses
        
        weighted_clauses = []
        
        for i, clause in enumerate(clauses):
            weighted_clause = clause.copy()
            
            # Apply weighting based on position
            if i in contrast_info['pre_contrast_indices']:
                # Reduce pre-contrast importance
                weighted_clause['confidence'] *= 0.7
                weighted_clause['contrast_position'] = 'pre'
            elif i in contrast_info['post_contrast_indices']:
                # Boost post-contrast importance
                weighted_clause['confidence'] *= 1.3
                # Cap at 0.99 to stay realistic
                weighted_clause['confidence'] = min(0.99, weighted_clause['confidence'])
                weighted_clause['contrast_position'] = 'post'
            
            weighted_clauses.append(weighted_clause)
        
        return weighted_clauses
    
    def fix_exhaustion_emotion(self, emotion: str, text: str) -> str:
        """
        RULE 2: Replace neutral with disappointment for exhaustion.
        
        Args:
            emotion: Original emotion
            text: Clause text
            
        Returns:
            Fixed emotion
        """
        if emotion == 'neutral' and self.contains_exhaustion(text):
            return 'disappointment'
        return emotion
    
    def deduplicate_clauses(self, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        RULE 3: Remove duplicate or substring clauses.
        
        Args:
            clauses: List of clause dictionaries
            
        Returns:
            Deduplicated clauses
        """
        if not clauses:
            return []
        
        cleaned = []
        seen_texts = set()
        
        for clause in clauses:
            text = clause.get('text', '').lower().strip()
            
            # Skip empty
            if not text:
                continue
            
            # Skip exact duplicates
            if text in seen_texts:
                continue
            
            # Check if substring of existing
            is_substring = False
            for existing in cleaned:
                existing_text = existing.get('text', '').lower().strip()
                if text in existing_text and text != existing_text:
                    is_substring = True
                    break
            
            if is_substring:
                continue
            
            # Check if existing clauses are substrings of this one
            to_remove = []
            for i, existing in enumerate(cleaned):
                existing_text = existing.get('text', '').lower().strip()
                if existing_text in text and existing_text != text:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                removed_text = cleaned.pop(i).get('text', '').lower().strip()
                seen_texts.discard(removed_text)
            
            cleaned.append(clause)
            seen_texts.add(text)
        
        return cleaned
    
    def apply_reasoning_fixes(
        self,
        original_text: str,
        raw_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main method: Fix reasoning errors in raw classification output.
        
        Args:
            original_text: Original input text
            raw_output: Raw emotion classification result
            
        Returns:
            Fixed emotion classification with corrected reasoning
        """
        # Extract components from raw output
        clauses = raw_output.get('clauses', [])
        
        # If no clauses, treat as single-clause
        if not clauses:
            emotion = raw_output.get('emotion', 'neutral')
            confidence = raw_output.get('confidence', 0.0)
            
            # Check if anticipatory
            is_anticipatory = self.is_anticipatory(original_text)
            if is_anticipatory:
                confidence *= 0.6  # Reduce by 40%
            
            # Fix exhaustion
            emotion = self.fix_exhaustion_emotion(emotion, original_text)
            
            return {
                'emotion_type': 'single',
                'primary_emotion': emotion,
                'secondary_emotions': [],
                'confidence': confidence,
                'explanation': self._generate_explanation(
                    original_text, emotion, confidence, is_anticipatory, []
                ),
                'clauses': [{
                    'text': original_text,
                    'emotion': emotion,
                    'confidence': confidence,
                    'emotion_source': 'anticipatory' if is_anticipatory else 'experienced'
                }]
            }
        
        # RULE 3: Deduplicate clauses
        clauses = self.deduplicate_clauses(clauses)
        
        # RULE 6 (NEW): Detect contrast and apply weighting
        contrast_info = self.detect_contrast_position(original_text, clauses)
        if contrast_info['contrast_detected']:
            clauses = self.apply_contrast_weighting(clauses, contrast_info)
        
        # Process each clause with rules
        processed_clauses = []
        experienced_emotions = []
        anticipatory_emotions = []
        
        for clause in clauses:
            text = clause.get('text', '')
            emotion = clause.get('emotion', 'neutral')
            confidence = clause.get('confidence', 0.0)
            
            # RULE 1: Check if anticipatory
            is_anticipatory = self.is_anticipatory(text)
            if is_anticipatory:
                confidence *= 0.6  # Reduce by 40%
                emotion_source = 'anticipatory'
                anticipatory_emotions.append((emotion, confidence))
            else:
                emotion_source = 'experienced'
                experienced_emotions.append((emotion, confidence))
            
            # RULE 2: Fix exhaustion
            emotion = self.fix_exhaustion_emotion(emotion, text)
            
            processed_clauses.append({
                'text': text,
                'emotion': emotion,
                'confidence': confidence,
                'emotion_source': emotion_source
            })
        
        # RULE 4: Determine primary emotion
        primary_emotion, emotion_type, secondary_emotions = self._select_primary_emotion(
            experienced_emotions,
            anticipatory_emotions,
            processed_clauses
        )
        
        # Calculate final confidence
        if experienced_emotions:
            final_confidence = max(conf for _, conf in experienced_emotions)
        elif anticipatory_emotions:
            final_confidence = max(conf for _, conf in anticipatory_emotions)
        else:
            final_confidence = 0.0
        
        # RULE 5: Honesty over certainty
        if final_confidence < 0.6:
            emotion_type = 'ambiguous'
        
        # Generate explanation
        explanation = self._generate_explanation(
            original_text,
            primary_emotion,
            final_confidence,
            len(anticipatory_emotions) > 0,
            processed_clauses,
            contrast_info
        )
        
        return {
            'emotion_type': emotion_type,
            'primary_emotion': primary_emotion,
            'secondary_emotions': secondary_emotions,
            'confidence': final_confidence,
            'explanation': explanation,
            'clauses': processed_clauses
        }
    
    def _select_primary_emotion(
        self,
        experienced: List[tuple],
        anticipatory: List[tuple],
        clauses: List[Dict[str, Any]]
    ) -> tuple:
        """
        RULE 4: Select primary emotion prioritizing experienced over anticipatory.
        
        Returns:
            Tuple of (primary_emotion, emotion_type, secondary_emotions)
        """
        # Prioritize experienced emotions
        if experienced:
            # Sort by confidence
            experienced_sorted = sorted(experienced, key=lambda x: x[1], reverse=True)
            primary = experienced_sorted[0][0]
            
            # Check for mixed emotions
            unique_emotions = set(e for e, _ in experienced)
            if len(unique_emotions) > 1:
                secondary = [e for e, _ in experienced_sorted[1:3] if e != primary]
                return (primary, 'mixed', secondary)
            else:
                return (primary, 'single', [])
        
        # Fall back to anticipatory if no experienced emotions
        elif anticipatory:
            anticipatory_sorted = sorted(anticipatory, key=lambda x: x[1], reverse=True)
            primary = anticipatory_sorted[0][0]
            return (primary, 'single', [])
        
        # Default
        return ('neutral', 'single', [])
    
    def _generate_explanation(
        self,
        text: str,
        emotion: str,
        confidence: float,
        has_anticipatory: bool,
        clauses: List[Dict[str, Any]],
        contrast_info: Dict[str, Any] = None
    ) -> str:
        """Generate explanation for the fixed reasoning."""
        parts = []
        
        # Base explanation
        if confidence < 0.6:
            parts.append(f"The text suggests {emotion}, but with low confidence ({confidence:.2f}).")
        else:
            parts.append(f"The primary emotion is {emotion} (confidence: {confidence:.2f}).")
        
        # Contrast dominance explanation
        if contrast_info and contrast_info.get('contrast_detected'):
            post_clauses = [clauses[i] for i in contrast_info.get('post_contrast_indices', [])]
            if post_clauses:
                post_emotions = set(c['emotion'] for c in post_clauses)
                parts.append(
                    f"Contrast detected: post-contrast emotion ({', '.join(post_emotions)}) "
                    f"weighted more heavily than pre-contrast."
                )
        
        # Anticipatory vs experienced
        if has_anticipatory:
            anticipatory_count = sum(
                1 for c in clauses 
                if c.get('emotion_source') == 'anticipatory'
            )
            experienced_count = len(clauses) - anticipatory_count
            
            if experienced_count > 0 and anticipatory_count > 0:
                parts.append(
                    f"The text contrasts expectation ({anticipatory_count} clause(s)) "
                    f"with reality ({experienced_count} clause(s))."
                )
            elif anticipatory_count == len(clauses):
                parts.append("All emotions expressed are anticipated, not currently experienced.")
        
        # Mixed emotions
        if len(clauses) > 1:
            unique_emotions = set(c['emotion'] for c in clauses)
            if len(unique_emotions) > 1:
                parts.append(f"Multiple emotions detected: {', '.join(unique_emotions)}.")
        
        return " ".join(parts)
