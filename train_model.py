#!/usr/bin/env python3
"""
F1 2025 Race Predictor - Model Training Script
Trains advanced ML models for race outcome prediction using recency-weighted features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import warnings
import logging
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRAINING_DATA_CSV = "training_data_weighted.csv"
MODEL_OUTPUT = "f1_prediction_model.joblib"
SCALER_OUTPUT = "f1_scaler.joblib"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
MODEL_METRICS_CSV = "model_metrics.csv"

def load_training_data():
    """Load the prepared training data"""
    print("📁 Loading training data...")
    
    try:
        df = pd.read_csv(TRAINING_DATA_CSV)
        print(f"  ✓ Loaded {len(df)} records with {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"  ❌ {TRAINING_DATA_CSV} not found. Run prepare_training_data.py first.")
        return None

def prepare_features_and_targets(df):
    """Prepare features and target variables for training"""
    print("\n🔧 Preparing features and targets...")
    
    # Define feature columns (excluding target variables and metadata)
    exclude_columns = [
        'round', 'raceName', 'circuit', 'date', 'driver', 'driverCode', 'constructor',
        'grid', 'position', 'points', 'status', 'laps', 'time', 'fastestLap',
        'qualyPosition', 'grid_diff', 'total_points', 'total_wins', 'total_podiums',
        'is_winner', 'is_podium', 'is_points'
    ]
    
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    print(f"  ✓ Using {len(feature_columns)} features for training")
    
    # Prepare feature matrix X
    X = df[feature_columns].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Convert any remaining non-numeric columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Prepare target variables
    y_winner = df['is_winner']
    y_podium = df['is_podium']
    y_points = df['is_points']
    
    print(f"  ✓ Feature matrix shape: {X.shape}")
    print(f"  ✓ Target distributions:")
    print(f"    - Winners: {y_winner.sum()} ({y_winner.mean()*100:.1f}%)")
    print(f"    - Podiums: {y_podium.sum()} ({y_podium.mean()*100:.1f}%)")
    print(f"    - Points: {y_points.sum()} ({y_points.mean()*100:.1f}%)")
    
    return X, y_winner, y_podium, y_points, feature_columns

def create_model_pipeline():
    """Create an advanced model pipeline with multiple algorithms"""
    print("\n🤖 Creating model pipeline...")
    
    # Base models with optimized hyperparameters
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        subsample=0.8,
        max_features='sqrt'
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        class_weight='balanced',
        max_features='sqrt',
        bootstrap=True
    )
    
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear'
    )
    
    # Create voting classifier with optimized weights
    voting_classifier = VotingClassifier(
        estimators=[
            ('gb', gb_model),
            ('rf', rf_model),
            ('lr', lr_model)
        ],
        voting='soft',
        weights=[0.5, 0.3, 0.2]
    )
    
    print("  ✓ Created voting classifier with GradientBoosting, RandomForest, and LogisticRegression")
    return voting_classifier

def train_model(X, y, feature_columns, model_name="winner"):
    """Train the prediction model"""
    print(f"\n🚀 Training {model_name.upper()} prediction model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    model = create_model_pipeline()
    
    print("  Training model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print(f"\n📊 Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.title()}: {value:.4f}")
    
    # Cross-validation
    print("\n🔄 Cross-validation scores:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"  ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model_type'] = model_name
    metrics_df['timestamp'] = datetime.now().isoformat()
    metrics_df.to_csv(MODEL_METRICS_CSV, index=False)
    
    # Feature importance (if available)
    try:
        if hasattr(model, 'named_estimators_') and 'gb' in model.named_estimators_:
            gb_model = model.named_estimators_['gb']
            if hasattr(gb_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': gb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                feature_importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
                print(f"\n  ✓ Feature importance saved to {FEATURE_IMPORTANCE_CSV}")
                print("  Top 10 features:")
                for i, row in feature_importance.head(10).iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"  ⚠️  Could not extract feature importance: {e}")
    
    return model, X_test, y_test, y_pred, y_prob, metrics

def save_model_and_scaler(model, X, feature_columns):
    """Save the trained model and scaler"""
    print("\n💾 Saving model and scaler...")
    
    # Save the model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"  ✓ Model saved to {MODEL_OUTPUT}")
    
    # Create and save a scaler for future use
    scaler = RobustScaler()
    scaler.fit(X)
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"  ✓ Scaler saved to {SCALER_OUTPUT}")
    
    # Save feature columns for reference
    feature_info = pd.DataFrame({
        'feature': feature_columns,
        'index': range(len(feature_columns))
    })
    feature_info.to_csv("feature_columns.csv", index=False)
    print(f"  ✓ Feature columns saved to feature_columns.csv")
    
    # Save model metadata
    metadata = {
        'model_type': 'ensemble_voting',
        'algorithms': ['gradient_boosting', 'random_forest', 'logistic_regression'],
        'weights': [0.5, 0.3, 0.2],
        'training_date': datetime.now().isoformat(),
        'feature_count': len(feature_columns),
        'training_samples': len(X)
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv("model_metadata.csv", index=False)
    print(f"  ✓ Model metadata saved to model_metadata.csv")

def test_model_predictions(model, X_test, y_test, y_pred, y_prob):
    """Test model predictions on sample data"""
    print("\n🧪 Testing model predictions...")
    
    # Create sample prediction dataframe
    test_results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'probability': y_prob
    })
    
    # Add some sample predictions
    print("  Sample predictions:")
    sample_size = min(10, len(test_results))
    sample = test_results.sample(sample_size, random_state=42)
    
    for i, row in sample.iterrows():
        status = "✓" if row['actual'] == row['predicted'] else "✗"
        print(f"    {status} Actual: {row['actual']}, Predicted: {row['predicted']}, Prob: {row['probability']:.3f}")
    
    # Save test results
    test_results.to_csv("test_predictions.csv", index=False)
    print(f"  ✓ Test predictions saved to test_predictions.csv")

def validate_model_files():
    """Validate that all required model files are present"""
    print("\n🔍 Validating model files...")
    
    required_files = [
        MODEL_OUTPUT,
        SCALER_OUTPUT,
        "feature_columns.csv",
        "model_metadata.csv"
    ]
    
    all_present = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} - Missing")
            all_present = False
    
    if all_present:
        print("  ✅ All model files validated successfully!")
    else:
        print("  ⚠️  Some model files are missing")
    
    return all_present

def main():
    """Main function to train the F1 prediction model"""
    print("🏎️  F1 2025 Prediction Model Training")
    print("=" * 50)
    
    # Load training data
    df = load_training_data()
    if df is None:
        return
    
    # Prepare features and targets
    X, y_winner, y_podium, y_points, feature_columns = prepare_features_and_targets(df)
    
    # Train winner prediction model
    print("\n🎯 Training WINNER prediction model...")
    winner_model, X_test, y_test, y_pred, y_prob, metrics = train_model(X, y_winner, feature_columns, "winner")
    
    # Save model and scaler
    save_model_and_scaler(winner_model, X, feature_columns)
    
    # Test predictions
    test_model_predictions(winner_model, X_test, y_test, y_pred, y_prob)
    
    # Validate model files
    validate_model_files()
    
    print("\n" + "=" * 50)
    print("✅ Model training complete!")
    print("\nFiles created:")
    print(f"  - {MODEL_OUTPUT} (trained model)")
    print(f"  - {SCALER_OUTPUT} (feature scaler)")
    print(f"  - {FEATURE_IMPORTANCE_CSV} (feature importance)")
    print(f"  - {MODEL_METRICS_CSV} (model performance metrics)")
    print(f"  - feature_columns.csv (feature mapping)")
    print(f"  - model_metadata.csv (model information)")
    print(f"  - test_predictions.csv (test results)")
    
    print("\nModel Performance Summary:")
    for metric, value in metrics.items():
        print(f"  {metric.title()}: {value:.4f}")
    
    print("\nNext steps:")
    print("  1. Review feature importance to understand key factors")
    print("  2. Use inference.py for making predictions")
    print("  3. Deploy to Hugging Face Hub")
    print("  4. Monitor model performance and retrain as needed")

if __name__ == "__main__":
    main()
