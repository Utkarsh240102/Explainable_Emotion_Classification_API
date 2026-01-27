"""
Explanation Generation Module.

Generates human-readable explanations for emotion predictions using
rule-based logic and keyword analysis.

NO external LLMs or APIs are used - explanations are fully deterministic.
"""

import re
from typing import Dict, List, Any


class ExplanationGenerator:
    """
    Generates human-readable explanations for emotion predictions.
    
    Uses rule-based logic and keyword matching without external APIs.
    """
    
    # Negation patterns
    NEGATION_WORDS = [
        'not', 'no', "isn't", "wasn't", "aren't", "weren't",
        "don't", "doesn't", "didn't", "never", "neither", "nor",
        "can't", "cannot", "won't", "wouldn't", "shouldn't", "couldn't",
        "hardly", "barely", "scarcely"
    ]
    
    # Mixed emotion indicators
    MIXED_EMOTION_PHRASES = [
        'part of me', 'kind of', 'sort of', 'somewhat', 'a bit',
        'mixed feelings', 'conflicted', 'both', 'but also',
        'on one hand', 'on the other hand', 'however', 'although',
        'even though', 'yet', 'still', 'nevertheless'
    ]
    
    # Conjunctions that indicate complexity
    CONJUNCTIONS = ['but', 'however', 'yet', 'although', 'though', 'while', 'whereas']
    
    # Keyword patterns for each emotion
    EMOTION_KEYWORDS = {
        'anger': [
            'angry', 'furious', 'mad', 'rage', 'hate', 'annoyed', 'irritated',
            'frustrated', 'upset', 'outraged', 'hostile', 'resentful', 'enraged',
            'livid', 'infuriated', 'awful', 'terrible', 'stupid',
            'idiot', 'worst', 'damn', 'hell'
        ],
        'disgust': [
            'disgusted', 'disgusting', 'gross', 'revolting', 'repulsive', 'nasty',
            'vile', 'repugnant', 'sick', 'yuck', 'eww', 'distasteful', 'foul'
        ],
        'fear': [
            'afraid', 'scared', 'fear', 'anxious', 'worried', 'nervous',
            'terrified', 'panic', 'frightened', 'concerned', 'alarmed',
            'uneasy', 'dread', 'horror', 'threat', 'danger', 'risk',
            'insecure', 'vulnerable', 'stressed', 'tense', 'uncertain'
        ],
        'joy': [
            'happy', 'joy', 'excited', 'delighted', 'wonderful', 'amazing',
            'great', 'fantastic', 'excellent', 'love', 'pleased', 'glad',
            'cheerful', 'thrilled', 'grateful', 'blessed', 'awesome', 'perfect',
            'beautiful', 'best', 'brilliant', 'celebrate', 'laugh', 'smile'
        ],
        'love': [
            'love', 'adore', 'cherish', 'affection', 'care', 'fond', 'devoted',
            'passionate', 'romantic', 'sweetheart', 'darling', 'treasure', 'heart'
        ],
        'neutral': [
            'okay', 'fine', 'normal', 'regular', 'usual', 'average',
            'typical', 'standard', 'ordinary', 'alright', 'so-so'
        ],
        'sadness': [
            'sad', 'depressed', 'unhappy', 'miserable', 'disappointed', 'lonely',
            'heartbroken', 'sorry', 'regret', 'cry', 'tears', 'grief', 'sorrow',
            'melancholy', 'gloomy', 'hopeless', 'despair', 'down', 'blue',
            'hurt', 'painful', 'loss', 'miss', 'broken'
        ],
        'surprise': [
            'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'startled',
            'unexpected', 'wow', 'omg', 'unbelievable', 'incredible', 'sudden',
            'caught off guard'
        ]
    }
    
    def __init__(self):
        """Initialize the explanation generator."""
        # Compile regex patterns for efficiency
        self.keyword_patterns = {
            emotion: self._compile_patterns(keywords)
            for emotion, keywords in self.EMOTION_KEYWORDS.items()
        }
    
    def _compile_patterns(self, keywords: List[str]) -> re.Pattern:
        """Compile keyword list into a single regex pattern."""
        pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _find_keywords(self, text: str, emotion: str) -> List[str]:
        """Find emotion-specific keywords in text."""
        pattern = self.keyword_patterns.get(emotion)
        if pattern:
            matches = pattern.findall(text)
            # Remove duplicates while preserving order
            return list(dict.fromkeys(matches))
        return []
    
    def _analyze_text_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features of the text."""
        features = {
            'length': len(text.split()),
            'has_exclamation': '!' in text,
            'has_question': '?' in text,
            'has_negation': any(word in text.lower() for word in ['not', 'no', "n't", 'never', 'neither']),
            'all_caps_words': len([w for w in text.split() if w.isupper() and len(w) > 1]),
            'has_ellipsis': '...' in text or '…' in text
        }
        return features
    
    def _get_confidence_descriptor(self, confidence: float) -> str:
        """Convert confidence score to descriptive text."""
        if confidence >= 0.9:
            return "very high confidence"
        elif confidence >= 0.75:
            return "high confidence"
        elif confidence >= 0.6:
            return "moderate confidence"
        elif confidence >= 0.45:
            return "moderate confidence"
        else:
            return "low confidence"
    
    def _detect_negation(self, text: str) -> bool:
        """Detect if text contains negation patterns."""
        text_lower = text.lower()
        return any(negation in text_lower for negation in self.NEGATION_WORDS)
    
    def _detect_mixed_emotion_phrases(self, text: str) -> List[str]:
        """Detect phrases that indicate mixed or complex emotions."""
        text_lower = text.lower()
        found_phrases = [phrase for phrase in self.MIXED_EMOTION_PHRASES if phrase in text_lower]
        return found_phrases
    
    def _detect_conjunctions(self, text: str) -> bool:
        """Detect conjunctions that suggest contrasting ideas."""
        text_lower = text.lower()
        return any(conj in text_lower for conj in self.CONJUNCTIONS)
    
    def _count_clauses(self, text: str) -> int:
        """Count number of clauses (rough estimate by commas and conjunctions)."""
        comma_count = text.count(',')
        conjunction_count = sum(1 for conj in self.CONJUNCTIONS if conj in text.lower())
        return comma_count + conjunction_count + 1
    
    def _analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for complexity indicators.
        
        Returns:
            Dictionary with complexity information
        """
        has_negation = self._detect_negation(text)
        mixed_phrases = self._detect_mixed_emotion_phrases(text)
        has_conjunctions = self._detect_conjunctions(text)
        num_clauses = self._count_clauses(text)
        
        # Calculate complexity score
        complexity_score = 0
        if has_negation:
            complexity_score += 1
        if mixed_phrases:
            complexity_score += 2  # Mixed phrases are strong indicators
        if has_conjunctions:
            complexity_score += 1
        if num_clauses > 2:
            complexity_score += 1
        
        return {
            'is_complex': complexity_score >= 2,
            'has_negation': has_negation,
            'mixed_phrases': mixed_phrases,
            'has_conjunctions': has_conjunctions,
            'num_clauses': num_clauses,
            'complexity_score': complexity_score
        }
    
    def _add_complexity_warning(
        self,
        base_explanation: str,
        complexity_info: Dict[str, Any]
    ) -> str:
        """Add warning/disclaimer for complex emotional text."""
        if not complexity_info['is_complex']:
            return base_explanation
        
        warning_parts = []
        
        # Build specific warnings based on what was detected
        detected_indicators = []
        
        if complexity_info['mixed_phrases']:
            phrases_str = "', '".join(complexity_info['mixed_phrases'][:2])
            detected_indicators.append(f"mixed emotional indicators ('{phrases_str}')")
        
        if complexity_info['has_negation']:
            detected_indicators.append("negation")
        
        if complexity_info['has_conjunctions']:
            detected_indicators.append("contrasting ideas")
        
        if detected_indicators:
            indicators_text = ", ".join(detected_indicators)
            warning = (
                f" ⚠️ Note: This text contains {indicators_text}, "
                f"suggesting conflicting or nuanced emotions that single-label "
                f"classification may oversimplify."
            )
            return base_explanation + warning
        
        return base_explanation
    
    def _explain_by_keywords(
        self,
        text: str,
        emotion: str,
        confidence: float,
        complexity_info: Dict[str, Any] = None
    ) -> str:
        """Generate explanation based on keyword presence."""
        keywords = self._find_keywords(text, emotion)
        features = self._analyze_text_features(text)
        confidence_desc = self._get_confidence_descriptor(confidence)
        
        explanation_parts = []
        
        # Adjust wording based on complexity
        expression_verb = "appears to express" if (complexity_info and complexity_info['is_complex']) else "expresses"
        
        # Base explanation with confidence
        explanation_parts.append(
            f"The text {expression_verb} {emotion} with {confidence_desc} ({confidence:.2f})."
        )
        
        # Keyword-based reasoning
        if keywords:
            if len(keywords) == 1:
                explanation_parts.append(
                    f"The word '{keywords[0]}' strongly indicates {emotion}."
                )
            elif len(keywords) <= 3:
                keyword_str = "', '".join(keywords)
                explanation_parts.append(
                    f"Words like '{keyword_str}' indicate {emotion}."
                )
            else:
                keyword_str = "', '".join(keywords[:3])
                explanation_parts.append(
                    f"Multiple {emotion}-related words (e.g., '{keyword_str}') are present."
                )
        
        # Linguistic feature-based reasoning
        if emotion == 'anger' and (features['has_exclamation'] or features['all_caps_words'] > 0):
            explanation_parts.append("Strong emphasis (exclamation marks or capitalization) suggests intensity.")
        
        elif emotion == 'joy' and features['has_exclamation']:
            explanation_parts.append("Exclamation marks often express excitement or happiness.")
        
        elif emotion == 'sadness' and features['has_ellipsis']:
            explanation_parts.append("The trailing thought pattern suggests contemplation or melancholy.")
        
        elif emotion == 'fear' and features['has_question']:
            explanation_parts.append("Questioning tone may indicate uncertainty or concern.")
        
        elif emotion == 'neutral' and not keywords:
            explanation_parts.append("The text lacks strong emotional indicators, suggesting a neutral tone.")
        
        # If no keywords found for the predicted emotion
        if not keywords and emotion != 'neutral':
            explanation_parts.append(
                f"The overall tone and context suggest {emotion}, though specific keywords are subtle."
            )
        
        return " ".join(explanation_parts)
    
    def _explain_by_distribution(
        self,
        text: str,
        emotion: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        complexity_info: Dict[str, any]
    ) -> str:
        """Generate explanation considering the full probability distribution."""
        # Get second highest probability
        sorted_probs = sorted(
            all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check if prediction is ambiguous
        if len(sorted_probs) > 1:
            second_emotion, second_prob = sorted_probs[1]
            
            # If second probability is close to first (within 0.15)
            if confidence - second_prob < 0.15:
                explanation = self._explain_by_keywords(text, emotion, confidence, complexity_info)
                explanation += f" Note: The text also shows elements of {second_emotion} ({second_prob:.2f}), suggesting mixed emotions."
                return explanation
        
        # Standard explanation
        return self._explain_by_keywords(text, emotion, confidence, complexity_info)
    
    def generate_explanation(
        self,
        text: str,
        emotion: str,
        confidence: float,
        all_probabilities: Dict[str, float]
    ) -> str:
        """
        Main method to generate explanation for a prediction.
        
        This orchestrates the explanation generation by:
        1. Analyzing the text for emotion-specific keywords
        2. Examining linguistic features (punctuation, emphasis, etc.)
        3. Considering the confidence level
        4. Checking for mixed emotions in the probability distribution
        5. Detecting complexity (negations, mixed signals, contradictions)
        6. Adding warnings for complex emotional text
        
        Args:
            text: Original input text
            emotion: Predicted emotion label
            confidence: Confidence score (0-1)
            all_probabilities: Dictionary mapping each emotion to its probability
            
        Returns:
            Human-readable explanation string
        """
        # Step 1: Analyze text complexity
        complexity_info = self._analyze_text_complexity(text)
        
        # Step 2: Generate base explanation (distribution-aware)
        base_explanation = self._explain_by_distribution(
            text,
            emotion,
            confidence,
            all_probabilities,
            complexity_info
        )
        
        # Step 3: Add complexity warning if needed
        final_explanation = self._add_complexity_warning(
            base_explanation,
            complexity_info
        )
        
        return final_explanation
    
    def generate_clause_level_explanation(
        self,
        text: str,
        clause_emotions: List[Dict[str, Any]],
        shift_analysis: Dict[str, Any],
        primary_emotion: str,
        emotion_type: str
    ) -> str:
        """
        Generate explanation for clause-level emotion analysis.
        Handles single, mixed, and ambiguous emotion types.
        
        Args:
            text: Original full text
            clause_emotions: List of emotion predictions for each clause
            shift_analysis: Analysis of emotion shifts
            primary_emotion: The primary/dominant emotion
            emotion_type: 'single', 'mixed', or 'ambiguous'
            
        Returns:
            Human-readable explanation
        """
        # Check for anticipatory clauses
        anticipatory_count = sum(
            1 for c in clause_emotions 
            if 'anticipatory' in str(c).lower() or 
               any(word in c.get('text', '').lower() for word in ['thought', 'expected', 'hoped'])
        )
        
        # Handle ambiguous emotions
        if emotion_type == 'ambiguous':
            explanation_parts = [
                f"The text expresses {primary_emotion}, but with low confidence across all clauses. "
                "The emotional state is ambiguous or subtle."
            ]
            
            if anticipatory_count > 0:
                explanation_parts.append(
                    f"{anticipatory_count} clause(s) express anticipated emotions rather than current feelings."
                )
            
            return " ".join(explanation_parts)
        
        # Handle single emotion
        if not shift_analysis['has_shift']:
            base = f"The text consistently expresses {primary_emotion} across all parts."
            
            if anticipatory_count > 0:
                return f"{base} Note: {anticipatory_count} clause(s) express anticipated emotions, reducing confidence."
            
            return base
        
        # Handle mixed emotions
        explanation_parts = []
        
        # Opening statement
        if shift_analysis['type'] == 'opposing':
            explanation_parts.append(
                f"The text contains opposing emotions. "
                f"The primary emotion is {primary_emotion}."
            )
        else:
            explanation_parts.append(
                f"The text expresses mixed emotions. "
                f"The dominant emotion is {primary_emotion}."
            )
        
        # Describe each clause
        clause_descriptions = []
        for i, clause_info in enumerate(clause_emotions, 1):
            clause_text = clause_info['text']
            emotion = clause_info['emotion']
            confidence = clause_info['confidence']
            
            # Check if anticipatory
            is_anticipatory = any(word in clause_text.lower() for word in ['thought', 'expected', 'hoped', 'would make me'])
            
            # Find keywords in this clause
            keywords = self._find_keywords(clause_text, emotion)
            
            prefix = "anticipated" if is_anticipatory else "experienced"
            
            if keywords:
                keyword_str = "', '".join(keywords[:2])
                clause_descriptions.append(
                    f"Clause {i} ('{clause_text[:40]}...') expresses {prefix} {emotion} "
                    f"(words: '{keyword_str}')"
                )
            else:
                clause_descriptions.append(
                    f"Clause {i} ('{clause_text[:40]}...') expresses {prefix} {emotion}"
                )
        
        # Add clause breakdowns
        explanation_parts.append(" ".join(clause_descriptions))
        
        # Add insight about emotion complexity
        unique_emotions = shift_analysis['unique_emotions']
        if len(unique_emotions) >= 3:
            explanation_parts.append(
                f"This demonstrates emotional complexity with {len(unique_emotions)} distinct emotions."
            )
        
        if anticipatory_count > 0:
            explanation_parts.append(
                f"Note: {anticipatory_count} clause(s) express anticipated rather than current emotions."
            )
        
        return " ".join(explanation_parts)
