# Emotion Classification Frontend

React frontend for the Explainable Emotion Classification API.

## Features

- 🎭 Real-time emotion classification
- 📊 Visual representation of emotion probabilities
- 📝 Detailed explanations for predictions
- 🔍 Clause-level emotion analysis
- 💡 Example texts for quick testing
- 📱 Responsive design

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure the API URL:
Edit `.env` file and set the backend API URL:
```
VITE_API_URL=http://localhost:8000
```

3. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The production build will be in the `dist` folder.

## Usage

1. Enter text in the input field or click an example
2. Click "Analyze Emotion" to get predictions
3. View the results including:
   - Primary emotion with confidence score
   - Detailed explanation
   - Clause-level analysis (for complex text)
   - All emotion probabilities with visualization

## Technologies Used

- React 18
- Vite
- Axios (API calls)
- Recharts (Data visualization)
- CSS3 (Styling)

## API Integration

The frontend connects to the FastAPI backend. Make sure the backend is running before starting the frontend.

Backend endpoints used:
- `POST /predict` - Emotion prediction
- `GET /health` - Health check
- `GET /emotions` - List of emotions

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── EmotionForm.jsx       # Input form component
│   │   ├── EmotionForm.css
│   │   ├── EmotionResults.jsx    # Results display component
│   │   └── EmotionResults.css
│   ├── services/
│   │   └── api.js                # API service layer
│   ├── App.jsx                   # Main app component
│   ├── App.css
│   ├── main.jsx                  # Entry point
│   └── index.css
├── index.html
├── vite.config.js
├── package.json
└── .env
```
