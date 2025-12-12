#!/usr/bin/env python3
"""
F1 2025 Race Predictor - Example Usage
Demonstrates how to use the trained model for predictions
"""

import json
from inference import F1Predictor
from huggingface_hub import hf_hub_download
import joblib

def example_basic_usage():
    """Example of basic model usage"""
    print("🏎️ F1 2025 Race Predictor - Basic Usage Example")
    print("=" * 50)
    
    # Initialize predictor (will try to load model from current directory)
    predictor = F1Predictor()
    
    if predictor.model is None:
        print("Model not loaded. Please ensure model files are in the current directory.")
        return
    
    # Example 1: Single driver prediction
    print("\nExample 1: Single Driver Prediction")
    print("-" * 30)
    
    driver_data = {
        'driver': 'Max Verstappen',
        'team': 'Red Bull Racing',
        'track_name': 'Monaco Grand Prix',
        'qualifying_position': 1,
        'grid_position': 1,
        'recent_form_score': 0.95,
        'track_dominance_score': 0.88,
        'season_points': 150,
        'season_wins': 8,
        'season_podiums': 12,
        'weather_temp': 24,
        'weather_rain_chance': 15,
        'weather_wind': 20,
        'team_performance_score': 0.92,
        'track_difficulty': 0.9,
        'track_type_score': 0.8,
        'driver_experience': 10,
        'team_development': 0.95,
        'ewma_points': 25.5,
        'recent_qualifying_avg': 1.2,
        'recent_race_avg': 1.8
    }
    
    prediction = predictor.predict_single_driver(driver_data)
    
    print(f"Driver: {prediction['driver']} ({prediction['team']})")
    print(f"Track: {driver_data['track_name']}")
    print(f"Win Probability: {prediction['win_probability']:.2%}")
    print(f"Podium Probability: {prediction['podium_probability']:.2%}")
    print(f"Expected Position: {prediction['expected_position']}")
    print(f"Confidence: {prediction['confidence']:.2%}")

def example_race_prediction():
    """Example of full race prediction"""
    print("\n🏁 Example 2: Full Race Prediction")
    print("-" * 30)
    
    predictor = F1Predictor()
    
    if predictor.model is None:
        print(" Model not loaded.")
        return
    
    # Example race data for British Grand Prix
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
                'recent_form': 0.92,
                'track_dominance': 0.85,
                'season_points': 150,
                'season_wins': 8,
                'season_podiums': 12,
                'team_performance': 0.92,
                'track_difficulty': 0.8,
                'track_type_score': 0.7,
                'experience_years': 10,
                'team_development': 0.95,
                'ewma_points': 25.5,
                'recent_qualifying_avg': 1.5,
                'recent_race_avg': 2.1
            },
            {
                'name': 'Lando Norris',
                'team': 'McLaren',
                'qualifying_position': 1,
                'recent_form': 0.88,
                'track_dominance': 0.75,
                'season_points': 120,
                'season_wins': 2,
                'season_podiums': 8,
                'team_performance': 0.85,
                'track_difficulty': 0.8,
                'track_type_score': 0.7,
                'experience_years': 6,
                'team_development': 0.90,
                'ewma_points': 20.2,
                'recent_qualifying_avg': 2.8,
                'recent_race_avg': 3.2
            },
            {
                'name': 'Charles Leclerc',
                'team': 'Ferrari',
                'qualifying_position': 3,
                'recent_form': 0.85,
                'track_dominance': 0.70,
                'season_points': 110,
                'season_wins': 1,
                'season_podiums': 6,
                'team_performance': 0.80,
                'track_difficulty': 0.8,
                'track_type_score': 0.7,
                'experience_years': 7,
                'team_development': 0.85,
                'ewma_points': 18.5,
                'recent_qualifying_avg': 3.2,
                'recent_race_avg': 4.1
            }
        ]
    }
    
    predictions = predictor.predict_race(race_data)
    
    print(f"Race: {race_data['track_name']}")
    print(f"Weather: {race_data['weather']['temperature']}°C, {race_data['weather']['rain_chance']}% rain chance")
    print("\nPredictions:")
    print("-" * 80)
    print(f"{'Pos':<3} {'Driver':<20} {'Team':<15} {'Win %':<8} {'Podium %':<10} {'Expected':<10}")
    print("-" * 80)
    
    for i, pred in enumerate(predictions, 1):
        print(f"{i:<3} {pred['driver']:<20} {pred['team']:<15} "
              f"{pred['win_probability']:<8.1%} {pred['podium_probability']:<10.1%} "
              f"{pred['expected_position']:<10}")

def example_huggingface_usage():
    """Example of using the model from Hugging Face Hub"""
    print("\nExample 3: Hugging Face Hub Usage")
    print("-" * 30)
    
    try:
        # Download model from Hugging Face Hub
        print("Downloading model from Hugging Face Hub...")
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
        
        print("Model loaded successfully from Hugging Face Hub!")
        
        # Create predictor with loaded model
        predictor = F1Predictor()
        predictor.model = model
        predictor.scaler = scaler
        
        # Make a prediction
        driver_data = {
            'driver': 'Lewis Hamilton',
            'team': 'Ferrari',
            'track_name': 'Italian Grand Prix',
            'qualifying_position': 4,
            'recent_form_score': 0.78,
            'season_points': 85,
            'weather_temp': 28,
            'weather_rain_chance': 5
        }
        
        prediction = predictor.predict_single_driver(driver_data)
        
        print(f"\nPrediction for {prediction['driver']}:")
        print(f"Win Probability: {prediction['win_probability']:.2%}")
        print(f"Podium Probability: {prediction['podium_probability']:.2%}")
        
    except Exception as e:
        print(f"Error loading from Hugging Face Hub: {e}")
        print("Note: Replace 'your-username' with your actual Hugging Face username")

def example_api_usage():
    """Example of API-style usage"""
    print("\n🌐 Example 4: API-Style Usage")
    print("-" * 30)
    
    from inference import predict_race_outcome
    
    predictor = F1Predictor()
    
    if predictor.model is None:
        print("Model not loaded.")
        return
    
    # Simple API call
    features = {
        'driver': 'Oscar Piastri',
        'team': 'McLaren',
        'track_name': 'Australian Grand Prix',
        'qualifying_position': 5,
        'recent_form_score': 0.82,
        'track_dominance_score': 0.65,
        'season_points': 45,
        'season_wins': 0,
        'season_podiums': 2,
        'weather_temp': 22,
        'weather_rain_chance': 10,
        'weather_wind': 15
    }
    
    prediction = predict_race_outcome(features, predictor.model, predictor.scaler)
    
    print(f"API Response for {prediction['driver']}:")
    print(json.dumps(prediction, indent=2))

def main():
    """Run all examples"""
    print(" F1 2025 Race Predictor - Complete Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_race_prediction()
    example_huggingface_usage()
    example_api_usage()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Customize the examples for your specific use case")
    print("2. Integrate with your application")
    print("3. Deploy to production")
    print("4. Monitor model performance")

if __name__ == "__main__":
    main()
