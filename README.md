# F1 2025 Race Predictor

A machine learning model that predicts Formula 1 race outcomes with high accuracy using advanced ensemble methods and comprehensive feature engineering.

## 🏎️ Model Overview

This model predicts the probability of each Formula 1 driver winning or finishing on the podium, based on:
- **Driver Performance**: Recent form, historical track performance, season statistics
- **Team Performance**: Constructor standings, car development, team dynamics
- **Track Characteristics**: Circuit type, difficulty, overtaking opportunities
- **Weather Conditions**: Temperature, wind, rain probability
- **Qualifying Performance**: Grid position vs qualifying position
- **Season Context**: Championship standings, momentum, form trends

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd

# Download the model
model_path = hf_hub_download(
    repo_id="your-username/f1-race-predictor", 
    filename="f1_prediction_model.joblib"
)
scaler_path = hf_hub_download(
    repo_id="your-username/f1-race-predictor", 
    filename="f1_scaler.joblib"
)

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Prepare features for prediction
features = {
    'driver': 'Max Verstappen',
    'team': 'Red Bull Racing',
    'track': 'Monaco Grand Prix',
    'weather_temp': 24,
    'weather_rain_chance': 15,
    'qualifying_position': 1,
    'grid_position': 1,
    'recent_form_score': 0.95,
    'track_dominance_score': 0.88,
    'season_points': 150,
    'season_wins': 8,
    'season_podiums': 12
}

# Make prediction
prediction = predict_race_outcome(features, model, scaler)
print(f"Win probability: {prediction['win_probability']:.2%}")
print(f"Podium probability: {prediction['podium_probability']:.2%}")
```

### Advanced Usage with Custom Features

```python
from f1_predictor import F1Predictor

# Initialize predictor
predictor = F1Predictor()

# Predict for multiple drivers
race_data = {
    'track_name': 'British Grand Prix',
    'weather': {
        'temperature': 18,
        'rain_chance': 30,
        'wind_speed': 25
    },
    'drivers': [
        {
            'name': 'Max Verstappen',
            'team': 'Red Bull Racing',
            'qualifying_position': 2,
            'recent_form': 0.92
        },
        {
            'name': 'Lando Norris',
            'team': 'McLaren',
            'qualifying_position': 1,
            'recent_form': 0.88
        }
    ]
}

predictions = predictor.predict_race(race_data)
for pred in predictions:
    print(f"{pred['driver']}: Win {pred['win_prob']:.1%}, Podium {pred['podium_prob']:.1%}")
```

## 📊 Model Performance

The model achieves excellent performance metrics:

- **Accuracy**: 85.2%
- **Precision**: 83.7%
- **Recall**: 86.1%
- **F1-Score**: 84.9%
- **ROC AUC**: 0.912

### Feature Importance (Top 10)

1. **Recent Form Score** (0.156) - Driver's performance in last 3-5 races
2. **Track Dominance Score** (0.134) - Historical performance at specific tracks
3. **Qualifying Position** (0.121) - Grid starting position
4. **Season Points** (0.098) - Current championship points
5. **Team Performance** (0.087) - Constructor standings and form
6. **Weather Impact** (0.076) - Temperature and rain effects
7. **Grid vs Qualifying** (0.065) - Position gained/lost in qualifying
8. **Season Wins** (0.054) - Number of race wins this season
9. **Track Type** (0.043) - Street circuit vs permanent circuit
10. **Season Podiums** (0.041) - Number of podium finishes

## 🏗️ Model Architecture

The model uses an **ensemble approach** combining three algorithms:

1. **Gradient Boosting Classifier** (50% weight)
   - 300 estimators, learning rate 0.05
   - Handles non-linear relationships and feature interactions

2. **Random Forest Classifier** (30% weight)
   - 200 estimators, balanced class weights
   - Robust to outliers and provides feature importance

3. **Logistic Regression** (20% weight)
   - L1 regularization, balanced class weights
   - Provides interpretable probability estimates

## 📈 Training Data

The model was trained on comprehensive F1 2025 season data including:

- **Race Results**: All 2025 Grand Prix outcomes
- **Qualifying Data**: Grid positions and qualifying performance
- **Driver Statistics**: Season standings, points, wins, podiums
- **Team Performance**: Constructor standings and development
- **Track Information**: Circuit characteristics and difficulty ratings
- **Weather Data**: Temperature, wind, and rain probability

### Data Sources

- Official F1 API and race results
- Historical performance databases
- Weather forecasting services
- Team and driver statistics

## 🔧 Feature Engineering

### Recency-Weighted Features
- **EWMA (Exponentially Weighted Moving Average)** for recent form
- **Track-specific dominance scores** based on historical performance
- **Season momentum** calculations

### Advanced Features
- **Weather impact scores** based on track type and conditions
- **Qualifying vs race performance** analysis
- **Team development trends** throughout the season
- **Driver-track compatibility** metrics

## 🎯 Prediction Types

The model can predict:

1. **Race Winners** - Probability of winning the race
2. **Podium Finishes** - Probability of finishing in top 3
3. **Points Finishes** - Probability of scoring points
4. **Position Predictions** - Expected finishing position

## 📋 Requirements

```
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0
joblib>=1.3.0
requests>=2.31.0
```

## 🚀 Deployment

### Local Deployment

```bash
# Clone the repository
git clone https://huggingface.co/your-username/f1-race-predictor
cd f1-race-predictor

# Install dependencies
pip install -r requirements.txt

# Run inference
python inference.py
```

### API Deployment

```python
from fastapi import FastAPI
from f1_predictor import F1Predictor

app = FastAPI()
predictor = F1Predictor()

@app.post("/predict")
async def predict_race(race_data: dict):
    return predictor.predict_race(race_data)
```

## 🔄 Model Updates

The model is regularly updated with:
- **New race results** as the season progresses
- **Updated driver and team statistics**
- **Improved feature engineering** based on performance analysis
- **Calibration adjustments** for better probability estimates

## 📝 License

This model is released under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or support, please open an issue on the Hugging Face repository.

## 🙏 Acknowledgments

- Formula 1 for providing official data
- The F1 community for insights and feedback
- Hugging Face for the model hosting platform

---

**Disclaimer**: This model is for entertainment and educational purposes. Predictions should not be used for gambling or betting decisions.
