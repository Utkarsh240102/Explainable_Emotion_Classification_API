"""
Clause-level Emotion Analysis Module.

This module splits complex sentences into clauses and analyzes
emotions at the clause level for better explainability.
"""

import re
from typing import List, Dict, Tuple, Any


class ClauseAnalyzer:
    """
    Analyzes text at the clause level to detect mixed or conflicting emotions.
    """
    
    # Conjunctions that typically separate contrasting ideas
    CONTRASTING_CONJUNCTIONS = [
        'but', 'however', 'yet', 'although', 'though', 'while',
        'whereas', 'nevertheless', 'nonetheless'
    ]
    
    # Phrases that indicate split perspectives
    SPLIT_PHRASES = [
        'part of me', 'on one hand', 'on the other hand',
        'at the same time', 'in contrast', 'conversely'
    ]
    
    # Coordinating conjunctions
    COORDINATING_CONJUNCTIONS = ['and', 'or', 'nor']
    
    # Anticipatory/expectation indicators
    ANTICIPATORY_WORDS = [
        'thought', 'expected', 'hoped', 'assumed', 'believed', 
        'would make me', 'should make me', 'will make me',
        'expecting', 'hoping', 'assuming', 'thinking'
    ]
    
    # Emotional exhaustion words
    EXHAUSTION_WORDS = [
        'tired', 'drained', 'exhausted', 'burned out', 'burnt out',
        'worn out', 'weary', 'fatigued', 'depleted'
    ]
    
    def __init__(self):
        """Initialize the clause analyzer."""
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for clause splitting."""
        # Pattern for contrasting conjunctions
        contrast_words = '|'.join(self.CONTRASTING_CONJUNCTIONS)
        self.contrast_pattern = re.compile(
            r'\b(' + contrast_words + r')\b',
            re.IGNORECASE
        )
        
        # Pattern for split phrases
        split_phrases = '|'.join(re.escape(phrase) for phrase in self.SPLIT_PHRASES)
        self.split_phrase_pattern = re.compile(
            r'(' + split_phrases + r')',
            re.IGNORECASE
        )
    
    def should_analyze_clauses(self, text: str) -> bool:
        """
        Determine if text should be analyzed at clause level.
        
        Args:
            text: Input text
            
        Returns:
            True if clause-level analysis is recommended
        """
        # Check for contrasting conjunctions
        if self.contrast_pattern.search(text):
            return True
        
        # Check for split perspective phrases
        if self.split_phrase_pattern.search(text):
            return True
        
        # Check for multiple clauses with commas + coordinating conjunctions
        if ',' in text and any(conj in text.lower() for conj in self.COORDINATING_CONJUNCTIONS):
            # More than 2 clauses suggest complexity
            clause_count = text.count(',') + 1
            if clause_count > 2:
                return True
        
        return False
    
    def _clean_clauses(self, clauses: List[str], min_words: int = 3) -> List[str]:
        """
        Remove duplicate, fragmentary, or repeated clauses.
        
        Rules:
        1. Remove exact duplicates
        2. Remove clauses that are substrings of longer clauses
        3. Remove clauses shorter than min_words meaningful words
        4. Preserve original order
        
        Args:
            clauses: List of clause strings to clean
            min_words: Minimum number of meaningful words (default 3)
        """
        if not clauses:
            return []
        
        cleaned = []
        seen = set()
        
        for clause in clauses:
            clause_lower = clause.lower().strip()
            
            # Skip empty
            if not clause_lower:
                continue
            
            # Count meaningful words (exclude very short words)
            words = [w for w in clause_lower.split() if len(w) > 2]
            if len(words) < min_words:
                continue
            
            # Check for exact duplicates
            if clause_lower in seen:
                continue
            
            # Check if this clause is a substring of any already added clause
            is_substring = False
            for existing in cleaned:
                if clause_lower in existing.lower() and clause_lower != existing.lower():
                    is_substring = True
                    break
            
            if is_substring:
                continue
            
            # Check if any existing clause is a substring of this one
            to_remove = []
            for i, existing in enumerate(cleaned):
                if existing.lower() in clause_lower and existing.lower() != clause_lower:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                cleaned.pop(i)
            
            cleaned.append(clause)
            seen.add(clause_lower)
        
        return cleaned
    
    def _is_anticipatory(self, clause: str) -> bool:
        """
        Detect if clause expresses expectation/anticipation rather than current emotion.
        
        Returns True if anticipatory, False if experienced.
        """
        clause_lower = clause.lower()
        return any(word in clause_lower for word in self.ANTICIPATORY_WORDS)
    
    def _contains_exhaustion(self, clause: str) -> bool:
        """
        Detect emotional exhaustion words that should NOT be labeled as neutral.
        
        Returns True if contains exhaustion words.
        """
        clause_lower = clause.lower()
        return any(word in clause_lower for word in self.EXHAUSTION_WORDS)
    
    def split_into_clauses(self, text: str) -> List[str]:
        """
        Split text into meaningful clauses and clean them.
        
        Args:
            text: Input text to split
            
        Returns:
            List of cleaned clause strings
        """
        clauses = []
        
        # Strategy 0: Handle "part of me X and part of me Y" pattern specifically
        if 'part of me' in text.lower():
            # Count occurrences of "part of me"
            count = text.lower().count('part of me')
            if count >= 2:
                # Split on "and" to separate the two parts
                if ' and ' in text.lower():
                    parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        # Both parts should contain "part of me"
                        clauses = [p.strip() for p in parts if p.strip()]
                        # Don't clean these - they're intentionally parallel structures
                        if len(clauses) == 2:
                            return clauses
        
        # Strategy 1: Split on contrasting conjunctions
        if self.contrast_pattern.search(text):
            parts = self.contrast_pattern.split(text)
            # Rejoin conjunction with following part
            for i in range(0, len(parts) - 1, 2):
                clause = parts[i].strip()
                if clause:
                    clauses.append(clause)
                # Add conjunction with next part
                if i + 2 < len(parts):
                    conjunction = parts[i + 1]
                    next_part = parts[i + 2].strip()
                    clauses.append(f"{conjunction} {next_part}")
            # Add last part if exists
            if len(parts) % 2 == 1 and parts[-1].strip():
                clauses.append(parts[-1].strip())
            
            cleaned = self._clean_clauses([c for c in clauses if c])
            # If cleaning resulted in empty list, return whole text
            if not cleaned:
                return [text.strip()]
            return cleaned
        
        # Strategy 2: Split on split phrases
        if self.split_phrase_pattern.search(text):
            parts = self.split_phrase_pattern.split(text)
            for i, part in enumerate(parts):
                part = part.strip()
                if part and part.lower() not in [p.lower() for p in self.SPLIT_PHRASES]:
                    # Include the split phrase with the clause
                    if i > 0 and i-1 < len(parts) and parts[i-1].lower() in [p.lower() for p in self.SPLIT_PHRASES]:
                        clauses.append(f"{parts[i-1]} {part}")
                    else:
                        clauses.append(part)
            
            cleaned = self._clean_clauses([c for c in clauses if c])
            # If cleaning resulted in empty list, return whole text
            if not cleaned:
                return [text.strip()]
            return cleaned
        
        # Strategy 3: Split on commas with coordinating conjunctions
        if ',' in text:
            parts = [p.strip() for p in text.split(',')]
            return self._clean_clauses([p for p in parts if p])
        
        # Default: return whole text as single clause
        return [text.strip()]
    
    def detect_emotion_shift(
        self,
        clause_emotions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect if there's an emotion shift across clauses.
        
        Args:
            clause_emotions: List of emotion predictions for each clause
            
        Returns:
            Analysis of emotion shifts
        """
        if len(clause_emotions) < 2:
            return {
                'has_shift': False,
                'type': 'single'
            }
        
        # Get unique emotions
        unique_emotions = set(c['emotion'] for c in clause_emotions)
        
        # Check for opposing emotions
        opposing_pairs = [
            ('joy', 'sadness'),
            ('love', 'anger'),
            ('excitement', 'disappointment'),
            ('optimism', 'fear'),
            ('admiration', 'disgust')
        ]
        
        has_opposing = False
        for e1, e2 in opposing_pairs:
            if e1 in unique_emotions and e2 in unique_emotions:
                has_opposing = True
                break
        
        return {
            'has_shift': len(unique_emotions) > 1,
            'type': 'opposing' if has_opposing else 'mixed',
            'unique_emotions': list(unique_emotions),
            'emotion_count': len(unique_emotions)
        }
    
    def get_primary_emotion(
        self,
        clause_emotions: List[Dict[str, Any]]
    ) -> Tuple[str, float, str, List[str]]:
        """
        Determine the primary emotion from clause-level analysis.
        Implements conservative confidence and handles anticipatory/exhaustion cases.
        
        Args:
            clause_emotions: List of emotion predictions for each clause
            
        Returns:
            Tuple of (primary_emotion, confidence, emotion_type, primary_emotions)
        """
        if not clause_emotions:
            return ("neutral", 0.0, "single", [])
        
        # If single clause
        if len(clause_emotions) == 1:
            clause = clause_emotions[0]
            confidence = clause['confidence']
            emotion = clause['emotion']
            text = clause.get('text', '')
            
            # Check if anticipatory - reduce confidence by 40%
            if self._is_anticipatory(text):
                confidence *= 0.6
            
            # Check for exhaustion words - prefer disappointment over neutral
            if emotion == 'neutral' and self._contains_exhaustion(text):
                emotion = 'disappointment'
                confidence *= 0.8
            
            return (emotion, confidence, "single", [])
        
        # For multiple clauses - calculate weighted emotions
        emotion_scores = {}
        anticipatory_count = 0
        
        for clause_data in clause_emotions:
            emotion = clause_data['emotion']
            confidence = clause_data['confidence']
            text = clause_data.get('text', '')
            
            # Reduce confidence for anticipatory clauses
            if self._is_anticipatory(text):
                confidence *= 0.6
                anticipatory_count += 1
            
            # Handle exhaustion words
            if emotion == 'neutral' and self._contains_exhaustion(text):
                emotion = 'disappointment'
                confidence *= 0.8
            
            if emotion not in emotion_scores:
                emotion_scores[emotion] = []
            emotion_scores[emotion].append(confidence)
        
        # Average confidence for each emotion
        emotion_avg = {
            emotion: sum(scores) / len(scores)
            for emotion, scores in emotion_scores.items()
        }
        
        # Check for opposing emotions
        opposing_pairs = [
            ('joy', 'sadness'),
            ('joy', 'disappointment'),
            ('love', 'anger'),
            ('excitement', 'disappointment'),
            ('optimism', 'fear'),
            ('admiration', 'disgust')
        ]
        
        unique_emotions = list(emotion_avg.keys())
        has_opposing = False
        for e1, e2 in opposing_pairs:
            if e1 in unique_emotions and e2 in unique_emotions:
                has_opposing = True
                break
        
        # Check if any emotion was implicit (from rule)
        has_implicit = any(c.get('is_implicit', False) for c in clause_emotions)
        
        # Sort emotions by confidence
        sorted_emotions = sorted(emotion_avg.items(), key=lambda x: x[1], reverse=True)
        max_confidence = sorted_emotions[0][1]
        
        # Determine emotion type and primary emotion
        if has_opposing:
            emotion_type = 'opposing'
            # For opposing emotions, use "conflicted" as summary with averaged confidence
            avg_confidence = sum(emotion_avg.values()) / len(emotion_avg)
            primary_emotion = 'conflicted'
            primary_emotions = [e[0] for e in sorted_emotions[:2]]  # Top 2 opposing emotions
        elif max_confidence < 0.6:
            emotion_type = 'ambiguous'
            # Weak/unclear emotions - keep "ambiguous"
            primary_emotion = 'ambiguous'
            primary_emotions = [e[0] for e in sorted_emotions[:2]]  # Top 2 emotions
            avg_confidence = max_confidence
        elif len(unique_emotions) == 1 or max_confidence >= 0.7:
            emotion_type = 'single'
            primary_emotion = sorted_emotions[0][0]
            primary_emotions = []
            avg_confidence = max_confidence
        else:
            emotion_type = 'mixed'
            primary_emotion = sorted_emotions[0][0]
            primary_emotions = [e[0] for e in sorted_emotions[1:3]]  # Next 2 emotions
            avg_confidence = max_confidence
        
        return (primary_emotion, avg_confidence, emotion_type, primary_emotions)
