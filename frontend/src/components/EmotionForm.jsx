import { useState } from 'react';
import { predictEmotion } from '../services/api';
import './EmotionForm.css';

function EmotionForm({ onPredictionSuccess, onPredictionError, onLoadingChange, onReset }) {
  const [text, setText] = useState('');
  const [examples] = useState([
    "I just got promoted at work! This is the best day ever!",
    "I'm so worried about the upcoming exam, I can't stop thinking about it.",
    "I can't believe they lied to me. I'm absolutely furious!",
    "My best friend moved away. I miss them so much.",
    "The weather is nice today."
  ]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      onPredictionError('Please enter some text to analyze');
      return;
    }

    onLoadingChange(true);
    onReset();

    try {
      const result = await predictEmotion(text);
      onPredictionSuccess(result);
    } catch (err) {
      onPredictionError(err.message || 'Failed to analyze emotion. Please try again.');
    } finally {
      onLoadingChange(false);
    }
  };

  const handleExampleClick = (example) => {
    setText(example);
  };

  const handleClear = () => {
    setText('');
    onReset();
  };

  return (
    <div className="emotion-form-card">
      <h2>Enter Text to Analyze</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste your text here..."
            rows="6"
            maxLength={512}
            className="text-input"
          />
          <div className="char-count">
            {text.length} / 512 characters
          </div>
        </div>

        <div className="button-group">
          <button type="submit" className="btn btn-primary">
            🔍 Analyze Emotion
          </button>
          <button type="button" onClick={handleClear} className="btn btn-secondary">
            Clear
          </button>
        </div>
      </form>

      <div className="examples-section">
        <h3>Try these examples:</h3>
        <div className="examples-list">
          {examples.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              className="example-btn"
              title={example}
            >
              {example.length > 60 ? example.substring(0, 60) + '...' : example}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default EmotionForm;
