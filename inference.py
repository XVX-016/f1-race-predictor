#!/usr/bin/env python3
"""
F1 2025 Race Predictor - Inference Script
Loads the trained model and provides prediction functionality
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1Predictor:
    """Main F1 prediction class for loading models and making predictions"""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Initialize the F1 predictor
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the feature scaler file
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Load model and scaler
        if model_path and scaler_path:
            self.load_model(model_path, scaler_path)
        else:
            # Try to load from default locations
            self._load_default_model()
    
    def _load_default_model(self):
        """Load model from default locations"""
        try:
            # Try current directory first
            model_path = Path("f1_prediction_model.joblib")
            scaler_path = Path("f1_scaler.joblib")
            
            if model_path.exists() and scaler_path.exists():
                self.load_model(str(model_path), str(scaler_path))
            else:
                logger.warning("Model files not found in current directory")
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """
        Load the trained model and scaler
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the feature scaler file
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load feature columns if available
            feature_file = Path("feature_columns.csv")
            if feature_file.exists():
                feature_df = pd.read_csv(feature_file)
                self.feature_columns = feature_df['feature'].tolist()
                logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            
            logger.info("✅ Model and scaler loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _prepare_features(self, driver_data: Dict) -> np.ndarray:
        """
        Prepare feature vector for prediction
        
        Args:
            driver_data: Dictionary containing driver and race information
            
        Returns:
            Feature vector as numpy array
        """
        if not self.feature_columns:
            raise ValueError("Feature columns not loaded. Please load the model first.")
        
        # Initialize feature vector with zeros
        features = np.zeros(len(self.feature_columns))
        
        # Map driver data to feature columns
        feature_mapping = {
            'recent_form_score': driver_data.get('recent_form_score', 0.5),
            'track_dominance_score': driver_data.get('track_dominance_score', 0.5),
            'qualifying_position': driver_data.get('qualifying_position', 10),
            'grid_position': driver_data.get('grid_position', 10),
            'season_points': driver_data.get('season_points', 0),
            'season_wins': driver_data.get('season_wins', 0),
            'season_podiums': driver_data.get('season_podiums', 0),
            'team_performance_score': driver_data.get('team_performance_score', 0.5),
            'weather_impact_score': driver_data.get('weather_impact_score', 1.0),
            'track_difficulty': driver_data.get('track_difficulty', 0.5),
            'grid_vs_qualifying_diff': driver_data.get('grid_position', 10) - driver_data.get('qualifying_position', 10),
            'ewma_points': driver_data.get('ewma_points', 0),
            'recent_qualifying_avg': driver_data.get('recent_qualifying_avg', 10),
            'recent_race_avg': driver_data.get('recent_race_avg', 10),
            'track_type_score': driver_data.get('track_type_score', 0.5),
            'weather_temp': driver_data.get('weather_temp', 24),
            'weather_rain_chance': driver_data.get('weather_rain_chance', 15),
            'weather_wind': driver_data.get('weather_wind', 20),
            'driver_experience': driver_data.get('driver_experience', 5),
            'team_development': driver_data.get('team_development', 0.5)
        }
        
        # Fill in features based on mapping
        for i, feature in enumerate(self.feature_columns):
            if feature in feature_mapping:
                features[i] = feature_mapping[feature]
        
        return features
    
    def predict_single_driver(self, driver_data: Dict) -> Dict:
        """
        Predict outcome for a single driver
        
        Args:
            driver_data: Dictionary containing driver and race information
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        try:
            # Prepare features
            features = self._prepare_features(driver_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            win_probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # Calculate podium probability (simplified - could be enhanced)
            podium_probability = min(win_probability * 2.5, 0.95)
            
            return {
                'driver': driver_data.get('driver', 'Unknown'),
                'team': driver_data.get('team', 'Unknown'),
                'win_probability': float(win_probability),
                'podium_probability': float(podium_probability),
                'expected_position': int(1 + (1 - win_probability) * 19),  # Rough estimate
                'confidence': float(abs(win_probability - 0.5) * 2)  # Higher confidence for extreme probabilities
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {driver_data.get('driver', 'Unknown')}: {e}")
            return {
                'driver': driver_data.get('driver', 'Unknown'),
                'team': driver_data.get('team', 'Unknown'),
                'win_probability': 0.05,
                'podium_probability': 0.15,
                'expected_position': 10,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_race(self, race_data: Dict) -> List[Dict]:
        """
        Predict outcomes for all drivers in a race
        
        Args:
            race_data: Dictionary containing race information and driver list
            
        Returns:
            List of prediction results for each driver
        """
        predictions = []
        
        # Extract race-level data
        track_name = race_data.get('track_name', 'Unknown Track')
        weather = race_data.get('weather', {})
        
        # Process each driver
        for driver_info in race_data.get('drivers', []):
            # Combine race data with driver data
            driver_data = {
                'driver': driver_info.get('name', 'Unknown'),
                'team': driver_info.get('team', 'Unknown'),
                'track_name': track_name,
                'weather_temp': weather.get('temperature', 24),
                'weather_rain_chance': weather.get('rain_chance', 15),
                'weather_wind': weather.get('wind_speed', 20),
                'qualifying_position': driver_info.get('qualifying_position', 10),
                'grid_position': driver_info.get('grid_position', driver_info.get('qualifying_position', 10)),
                'recent_form_score': driver_info.get('recent_form', 0.5),
                'track_dominance_score': driver_info.get('track_dominance', 0.5),
                'season_points': driver_info.get('season_points', 0),
                'season_wins': driver_info.get('season_wins', 0),
                'season_podiums': driver_info.get('season_podiums', 0),
                'team_performance_score': driver_info.get('team_performance', 0.5),
                'track_difficulty': driver_info.get('track_difficulty', 0.5),
                'track_type_score': driver_info.get('track_type_score', 0.5),
                'driver_experience': driver_info.get('experience_years', 5),
                'team_development': driver_info.get('team_development', 0.5),
                'ewma_points': driver_info.get('ewma_points', 0),
                'recent_qualifying_avg': driver_info.get('recent_qualifying_avg', 10),
                'recent_race_avg': driver_info.get('recent_race_avg', 10)
            }
            
            # Calculate weather impact
            weather_impact = self._calculate_weather_impact(weather, track_name)
            driver_data['weather_impact_score'] = weather_impact
            
            # Make prediction
            prediction = self.predict_single_driver(driver_data)
            predictions.append(prediction)
        
        # Sort by win probability (descending)
        predictions.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return predictions
    
    def _calculate_weather_impact(self, weather: Dict, track_name: str) -> float:
        """
        Calculate weather impact score based on track and conditions
        
        Args:
            weather: Weather conditions dictionary
            track_name: Name of the track
            
        Returns:
            Weather impact score (0.5-1.5)
        """
        impact = 1.0
        
        # Temperature effects
        temp = weather.get('temperature', 24)
        if temp > 30:
            impact *= 0.95  # Hot weather slightly reduces performance
        elif temp < 10:
            impact *= 0.97  # Cold weather slightly reduces performance
        
        # Rain effects
        rain_chance = weather.get('rain_chance', 15)
        if rain_chance > 50:
            impact *= 0.90  # Heavy rain significantly reduces performance
        elif rain_chance > 20:
            impact *= 0.95  # Light rain slightly reduces performance
        
        # Wind effects
        wind = weather.get('wind_speed', 20)
        if wind > 30:
            impact *= 0.93  # High winds reduce performance
        elif wind > 20:
            impact *= 0.97  # Moderate winds slightly reduce performance
        
        # Track-specific adjustments
        if 'Monaco' in track_name or 'Singapore' in track_name:
            if rain_chance > 30:
                impact *= 0.92  # Street circuits are more sensitive to rain
        
        return max(0.5, min(1.5, impact))

def predict_race_outcome(features: Dict, model, scaler) -> Dict:
    """
    Simple prediction function for basic usage
    
    Args:
        features: Feature dictionary
        model: Loaded model
        scaler: Loaded scaler
        
    Returns:
        Prediction results
    """
    predictor = F1Predictor()
    predictor.model = model
    predictor.scaler = scaler
    
    return predictor.predict_single_driver(features)

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 2025 Race Predictor')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--scaler', help='Path to scaler file')
    parser.add_argument('--driver', help='Driver name')
    parser.add_argument('--team', help='Team name')
    parser.add_argument('--track', help='Track name')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = F1Predictor(args.model, args.scaler)
    
    if predictor.model is None:
        print("❌ Model not loaded. Please provide model and scaler paths.")
        return
    
    # Example prediction
    if args.driver and args.team and args.track:
        driver_data = {
            'driver': args.driver,
            'team': args.team,
            'track_name': args.track,
            'qualifying_position': 1,
            'grid_position': 1,
            'recent_form_score': 0.9,
            'track_dominance_score': 0.8,
            'season_points': 100,
            'season_wins': 5,
            'season_podiums': 8,
            'weather_temp': 24,
            'weather_rain_chance': 15,
            'weather_wind': 20
        }
        
        prediction = predictor.predict_single_driver(driver_data)
        print(f"\n🏎️ Prediction for {prediction['driver']} ({prediction['team']})")
        print(f"Track: {args.track}")
        print(f"Win Probability: {prediction['win_probability']:.2%}")
        print(f"Podium Probability: {prediction['podium_probability']:.2%}")
        print(f"Expected Position: {prediction['expected_position']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
    else:
        print("✅ F1 Predictor loaded successfully!")
        print("Use --driver, --team, and --track arguments for predictions")

if __name__ == "__main__":
    main()
