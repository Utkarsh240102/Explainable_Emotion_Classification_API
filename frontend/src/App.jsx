import { useState } from 'react';
import EmotionForm from './components/EmotionForm';
import EmotionResults from './components/EmotionResults';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredictionSuccess = (result) => {
    setPrediction(result);
    setError(null);
  };

  const handlePredictionError = (err) => {
    setError(err);
    setPrediction(null);
  };

  const handleReset = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎭 Explainable Emotion Classification</h1>
        <p className="subtitle">AI-powered emotion analysis with detailed explanations</p>
      </header>

      <main className="app-main">
        <div className="container">
          <EmotionForm
            onPredictionSuccess={handlePredictionSuccess}
            onPredictionError={handlePredictionError}
            onLoadingChange={setLoading}
            onReset={handleReset}
          />

          {error && (
            <div className="error-message">
              <h3>⚠️ Error</h3>
              <p>{error}</p>
            </div>
          )}

          {prediction && !loading && (
            <EmotionResults prediction={prediction} />
          )}

          {loading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <p>Analyzing emotion...</p>
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Powered by RoBERTa GoEmotions Model</p>
      </footer>
    </div>
  );
}

export default App;
