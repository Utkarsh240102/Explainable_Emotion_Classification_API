import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import './EmotionResults.css';

const EMOTION_COLORS = {
  admiration: '#FF6B9D',
  amusement: '#FFA07A',
  anger: '#FF4444',
  annoyance: '#FF6347',
  approval: '#90EE90',
  caring: '#FFB6C1',
  confusion: '#DDA0DD',
  curiosity: '#87CEEB',
  desire: '#FF69B4',
  disappointment: '#A9A9A9',
  disapproval: '#8B4513',
  disgust: '#6B8E23',
  embarrassment: '#FFB347',
  excitement: '#FF8C00',
  fear: '#8B008B',
  gratitude: '#FFD700',
  grief: '#696969',
  joy: '#FFD700',
  love: '#FF1493',
  nervousness: '#9370DB',
  optimism: '#00CED1',
  pride: '#DAA520',
  realization: '#4682B4',
  relief: '#98FB98',
  remorse: '#BC8F8F',
  sadness: '#4169E1',
  surprise: '#FF6347',
  neutral: '#808080',
  conflicted: '#9370DB',
  ambiguous: '#A9A9A9'
};

function EmotionResults({ prediction }) {
  const { emotion, confidence, all_emotions, explanation, emotion_type, clauses, primary_emotions } = prediction;


  const getEmotionEmoji = (emotionName) => {
    const emojiMap = {
      admiration: '😍',
      amusement: '😄',
      anger: '😠',
      annoyance: '😒',
      approval: '👍',
      caring: '🤗',
      confusion: '😕',
      curiosity: '🤔',
      desire: '😏',
      disappointment: '😞',
      disapproval: '👎',
      disgust: '🤢',
      embarrassment: '😳',
      excitement: '🎉',
      fear: '😨',
      gratitude: '🙏',
      grief: '😭',
      joy: '😊',
      love: '❤️',
      nervousness: '😰',
      neutral: '😐',
      optimism: '🌟',
      pride: '😌',
      realization: '💡',
      relief: '😌',
      remorse: '😔',
      sadness: '😢',
      surprise: '😲',
      conflicted: '😐😊',
      ambiguous: '🤔'
    };
    return emojiMap[emotionName.toLowerCase()] || '😐';
  };

  return (
    <div className="emotion-results">
      {/* PRIMARY EMOTION SECTION */}
      <div className="primary-emotion-card">
        <div className="emotion-header">
          {emotion === 'conflicted' && primary_emotions && primary_emotions.length >= 2 ? (
            <div className="dual-emoji">
              <span className="emoji-large">{getEmotionEmoji(primary_emotions[0])}</span>
              <span className="vs-text">vs</span>
              <span className="emoji-large">{getEmotionEmoji(primary_emotions[1])}</span>
            </div>
          ) : (
            <span className="emoji-large">{getEmotionEmoji(emotion)}</span>
          )}
        </div>
        
        <h2 className="emotion-title">{emotion.toUpperCase()}</h2>
        
        {emotion === 'conflicted' && primary_emotions && primary_emotions.length > 0 && (
          <div className="conflicted-emotions">
            {primary_emotions.map((emo, idx) => (
              <span key={idx} className="conflict-tag">
                {emo}{idx < primary_emotions.length - 1 ? ' vs ' : ''}
              </span>
            ))}
          </div>
        )}
        
        <div className="confidence-container">
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ 
                width: `${confidence * 100}%`,
                background: EMOTION_COLORS[emotion] || '#667eea'
              }}
            />
          </div>
          <p className="confidence-value">{(confidence * 100).toFixed(1)}% confidence</p>
        </div>
      </div>

      {/* EXPLANATION SECTION */}
      <div className="explanation-card">
        <h3>📝 Explanation</h3>
        <p className="explanation-text">{explanation}</p>
        <div className="meta-info">
          <span className="emotion-type-badge">{emotion_type}</span>
          {primary_emotions && primary_emotions.length > 0 && emotion !== 'conflicted' && (
            <div className="other-emotions">
              <strong>Also detected:</strong>{' '}
              {primary_emotions.map((emo, idx) => (
                <span key={idx} className="emotion-chip">
                  {getEmotionEmoji(emo)} {emo}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* CLAUSE-LEVEL ANALYSIS */}
      {clauses && clauses.length > 0 && (
        <div className="clauses-card">
          <h3>📋 Clause-by-Clause Analysis</h3>
          
          {clauses.map((clause, index) => {
            // Prepare chart data for this clause
            const clauseChartData = clause.all_probabilities 
              ? Object.entries(clause.all_probabilities)
                  .map(([name, value]) => ({
                    name,
                    value: parseFloat((value * 100).toFixed(2))
                  }))
                  .filter(item => item.value > 2.0) // Show emotions > 2%
                  .sort((a, b) => b.value - a.value)
                  .slice(0, 5) // Top 5 emotions
              : [];

            return (
              <div key={index} className="clause-item">
                <div className="clause-header">
                  <div className="clause-emotion">
                    <span className="clause-emoji">{getEmotionEmoji(clause.emotion)}</span>
                    <span className="clause-emotion-name">{clause.emotion}</span>
                    <span className="clause-confidence">{(clause.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <p className="clause-text">"{clause.text}"</p>
                </div>
                
                {clauseChartData.length > 0 && (
                  <div className="clause-chart">
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={clauseChartData}
                          cx="50%"
                          cy="50%"
                          innerRadius={0}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, value }) => `${name}: ${value}%`}
                        >
                          {clauseChartData.map((entry, idx) => (
                            <Cell 
                              key={`cell-${idx}`} 
                              fill={EMOTION_COLORS[entry.name] || '#667eea'}
                            />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value}%`} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ALL EMOTIONS BAR CHART */}
      <div className="all-emotions-card">
        <h3>📊 All Emotion Probabilities</h3>
        <div className="emotions-list">
          {Object.entries(all_emotions)
            .sort((a, b) => b[1] - a[1])
            .map(([name, value]) => (
              <div key={name} className="emotion-bar-item">
                <div className="emotion-label">
                  <span className="emotion-icon">{getEmotionEmoji(name)}</span>
                  <span className="emotion-name">{name}</span>
                </div>
                <div className="bar-container">
                  <div 
                    className="bar-fill" 
                    style={{ 
                      width: `${value * 100}%`,
                      background: EMOTION_COLORS[name] || '#667eea'
                    }}
                  />
                </div>
                <span className="emotion-percent">{(value * 100).toFixed(1)}%</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

export default EmotionResults;
