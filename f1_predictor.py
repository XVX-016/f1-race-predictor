#!/usr/bin/env python3
"""
F1 2025 Race Predictor - Main Module
Simplified wrapper for easy imports and usage
"""

from .inference import F1Predictor, predict_race_outcome

__version__ = "1.0.0"
__author__ = "F1 Prediction Team"

# Export main classes and functions
__all__ = ['F1Predictor', 'predict_race_outcome']

# Convenience function for quick predictions
def quick_predict(driver_name: str, team: str, track: str, **kwargs):
    """
    Quick prediction function for simple use cases
    
    Args:
        driver_name: Name of the driver
        team: Team name
        track: Track name
        **kwargs: Additional driver/race parameters
        
    Returns:
        Prediction dictionary
    """
    predictor = F1Predictor()
    
    if predictor.model is None:
        raise ValueError("Model not loaded. Please ensure model files are available.")
    
    # Default values
    defaults = {
        'qualifying_position': 10,
        'grid_position': 10,
        'recent_form_score': 0.5,
        'track_dominance_score': 0.5,
        'season_points': 0,
        'season_wins': 0,
        'season_podiums': 0,
        'weather_temp': 24,
        'weather_rain_chance': 15,
        'weather_wind': 20
    }
    
    # Update with provided values
    defaults.update(kwargs)
    
    # Create driver data
    driver_data = {
        'driver': driver_name,
        'team': team,
        'track_name': track,
        **defaults
    }
    
    return predictor.predict_single_driver(driver_data)
