#!/usr/bin/env python3
"""
Test script for F1 2025 Race Predictor
"""

from inference import F1Predictor

def test_model_loading():
    """Test that the model loads correctly"""
    print("🧪 Testing model loading...")
    
    try:
        predictor = F1Predictor()
        if predictor.model is not None and predictor.scaler is not None:
            print("✅ Model and scaler loaded successfully!")
            return True
        else:
            print("❌ Model or scaler not loaded")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_single_prediction():
    """Test a single prediction"""
    print("\n🧪 Testing single prediction...")
    
    try:
        predictor = F1Predictor()
        
        # Test data for Max Verstappen
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
            'weather_wind': 20
        }
        
        prediction = predictor.predict_single_driver(driver_data)
        
        print(f"✅ Prediction successful!")
        print(f"Driver: {prediction['driver']}")
        print(f"Win Probability: {prediction['win_probability']:.2%}")
        print(f"Podium Probability: {prediction['podium_probability']:.2%}")
        print(f"Expected Position: {prediction['expected_position']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

def test_race_prediction():
    """Test full race prediction"""
    print("\n🧪 Testing race prediction...")
    
    try:
        predictor = F1Predictor()
        
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
                    'qualifying_position': 1,
                    'recent_form': 0.92,
                    'track_dominance': 0.85,
                    'season_points': 150,
                    'season_wins': 8,
                    'season_podiums': 12
                },
                {
                    'name': 'Lando Norris',
                    'team': 'McLaren',
                    'qualifying_position': 2,
                    'recent_form': 0.88,
                    'track_dominance': 0.75,
                    'season_points': 120,
                    'season_wins': 2,
                    'season_podiums': 8
                }
            ]
        }
        
        predictions = predictor.predict_race(race_data)
        
        print(f"✅ Race prediction successful!")
        print(f"Number of predictions: {len(predictions)}")
        
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['driver']}: Win {pred['win_probability']:.1%}, Podium {pred['podium_probability']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error making race prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("🏎️ F1 2025 Race Predictor - Model Testing")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_single_prediction,
        test_race_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the model files.")
    
    return passed == total

if __name__ == "__main__":
    main()
