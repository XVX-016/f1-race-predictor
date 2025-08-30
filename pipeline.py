#!/usr/bin/env python3
"""
F1 2025 Race Predictor - Hugging Face Pipeline
Compatible with Hugging Face inference API and widgets
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Any
import logging

logger = logging.getLogger(__name__)

class F1PredictionPipeline:
    """Hugging Face compatible pipeline for F1 race predictions"""
    
    def __init__(self, model_path: str = "f1_prediction_model.joblib", 
                 scaler_path: str = "f1_scaler.joblib"):
        """
        Initialize the F1 prediction pipeline
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the feature scaler file
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Load model and scaler
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load feature columns if available
            try:
                feature_df = pd.read_csv("feature_columns.csv")
                self.feature_columns = feature_df['feature'].tolist()
                logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            except FileNotFoundError:
                logger.warning("feature_columns.csv not found, using default features")
                self.feature_columns = [
                    'recent_form_score', 'track_dominance_score', 'qualifying_position',
                    'grid_position', 'season_points', 'season_wins', 'season_podiums',
                    'team_performance_score', 'weather_impact_score', 'track_difficulty',
                    'grid_vs_qualifying_diff', 'ewma_points', 'recent_qualifying_avg',
                    'recent_race_avg', 'track_type_score', 'weather_temp',
                    'weather_rain_chance', 'weather_wind', 'driver_experience', 'team_development'
                ]
            
            logger.info("✅ F1 Prediction Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def _prepare_features(self, inputs: Dict[str, Any]) -> np.ndarray:
        """
        Prepare feature vector from input dictionary
        
        Args:
            inputs: Dictionary containing driver and race information
            
        Returns:
            Feature vector as numpy array
        """
        if not self.feature_columns:
            raise ValueError("Feature columns not loaded")
        
        # Initialize feature vector with zeros
        features = np.zeros(len(self.feature_columns))
        
        # Default values for missing features
        defaults = {
            'recent_form_score': 0.5,
            'track_dominance_score': 0.5,
            'qualifying_position': 10,
            'grid_position': 10,
            'season_points': 0,
            'season_wins': 0,
            'season_podiums': 0,
            'team_performance_score': 0.5,
            'weather_impact_score': 1.0,
            'track_difficulty': 0.5,
            'grid_vs_qualifying_diff': 0,
            'ewma_points': 0,
            'recent_qualifying_avg': 10,
            'recent_race_avg': 10,
            'track_type_score': 0.5,
            'weather_temp': 24,
            'weather_rain_chance': 15,
            'weather_wind': 20,
            'driver_experience': 5,
            'team_development': 0.5
        }
        
        # Map inputs to features
        feature_mapping = {
            'driver': None,  # Not used in features
            'team': None,    # Not used in features
            'track_name': None,  # Not used in features
            'recent_form_score': inputs.get('recent_form_score', defaults['recent_form_score']),
            'track_dominance_score': inputs.get('track_dominance_score', defaults['track_dominance_score']),
            'qualifying_position': inputs.get('qualifying_position', defaults['qualifying_position']),
            'grid_position': inputs.get('grid_position', inputs.get('qualifying_position', defaults['grid_position'])),
            'season_points': inputs.get('season_points', defaults['season_points']),
            'season_wins': inputs.get('season_wins', defaults['season_wins']),
            'season_podiums': inputs.get('season_podiums', defaults['season_podiums']),
            'team_performance_score': inputs.get('team_performance_score', defaults['team_performance_score']),
            'weather_impact_score': inputs.get('weather_impact_score', defaults['weather_impact_score']),
            'track_difficulty': inputs.get('track_difficulty', defaults['track_difficulty']),
            'grid_vs_qualifying_diff': (inputs.get('grid_position', inputs.get('qualifying_position', 10)) - 
                                       inputs.get('qualifying_position', 10)),
            'ewma_points': inputs.get('ewma_points', defaults['ewma_points']),
            'recent_qualifying_avg': inputs.get('recent_qualifying_avg', defaults['recent_qualifying_avg']),
            'recent_race_avg': inputs.get('recent_race_avg', defaults['recent_race_avg']),
            'track_type_score': inputs.get('track_type_score', defaults['track_type_score']),
            'weather_temp': inputs.get('weather_temp', defaults['weather_temp']),
            'weather_rain_chance': inputs.get('weather_rain_chance', defaults['weather_rain_chance']),
            'weather_wind': inputs.get('weather_wind', defaults['weather_wind']),
            'driver_experience': inputs.get('driver_experience', defaults['driver_experience']),
            'team_development': inputs.get('team_development', defaults['team_development'])
        }
        
        # Fill in features based on mapping
        for i, feature in enumerate(self.feature_columns):
            if feature in feature_mapping and feature_mapping[feature] is not None:
                features[i] = feature_mapping[feature]
        
        return features
    
    def __call__(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Main prediction method compatible with Hugging Face pipeline
        
        Args:
            inputs: Single input dictionary or list of input dictionaries
            
        Returns:
            Prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded. Please initialize the pipeline first.")
        
        # Handle single input vs batch input
        if isinstance(inputs, dict):
            return self._predict_single(inputs)
        elif isinstance(inputs, list):
            return [self._predict_single(input_dict) for input_dict in inputs]
        else:
            raise ValueError("Inputs must be a dictionary or list of dictionaries")
    
    def _predict_single(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single input
        
        Args:
            inputs: Input dictionary
            
        Returns:
            Prediction results
        """
        try:
            # Prepare features
            features = self._prepare_features(inputs)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            win_probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # Calculate additional probabilities
            podium_probability = min(win_probability * 2.5, 0.95)
            points_probability = min(win_probability * 5.0, 0.98)
            
            # Calculate expected position
            expected_position = int(1 + (1 - win_probability) * 19)
            
            # Calculate confidence
            confidence = float(abs(win_probability - 0.5) * 2)
            
            return {
                'driver': inputs.get('driver', 'Unknown'),
                'team': inputs.get('team', 'Unknown'),
                'track': inputs.get('track_name', 'Unknown'),
                'win_probability': float(win_probability),
                'podium_probability': float(podium_probability),
                'points_probability': float(points_probability),
                'expected_position': expected_position,
                'confidence': confidence,
                'prediction_quality': 'high' if confidence > 0.3 else 'medium' if confidence > 0.1 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'driver': inputs.get('driver', 'Unknown'),
                'team': inputs.get('team', 'Unknown'),
                'track': inputs.get('track_name', 'Unknown'),
                'error': str(e),
                'win_probability': 0.05,
                'podium_probability': 0.15,
                'points_probability': 0.30,
                'expected_position': 10,
                'confidence': 0.0,
                'prediction_quality': 'error'
            }

# Convenience function for Hugging Face integration
def predict_f1_race(inputs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function for making F1 race predictions
    
    Args:
        inputs: Input data for prediction
        
    Returns:
        Prediction results
    """
    pipeline = F1PredictionPipeline()
    return pipeline(inputs)

# Example usage for Hugging Face
if __name__ == "__main__":
    # Example single prediction
    sample_input = {
        'driver': 'Max Verstappen',
        'team': 'Red Bull Racing',
        'track_name': 'Monaco Grand Prix',
        'qualifying_position': 1,
        'recent_form_score': 0.95,
        'track_dominance_score': 0.88,
        'season_points': 150,
        'season_wins': 8,
        'season_podiums': 12,
        'weather_temp': 24,
        'weather_rain_chance': 15
    }
    
    pipeline = F1PredictionPipeline()
    result = pipeline(sample_input)
    print("Sample prediction result:")
    print(result)
