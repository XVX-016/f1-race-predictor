#!/usr/bin/env python3
"""
F1 2025 Race Predictor - Complete Hugging Face Deployment Script
Automatically deploys the model to Hugging Face Hub with all necessary files
"""

import os
import subprocess
import sys
import json
import shutil
from pathlib import Path
import time

def print_step(step_num, title, description=""):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"🚀 STEP {step_num}: {title}")
    print(f"{'='*60}")
    if description:
        print(f"📝 {description}")
    print()

def check_prerequisites():
    """Check if all prerequisites are installed"""
    print_step(1, "Checking Prerequisites", "Verifying required tools are installed")
    
    # Check Python
    try:
        python_version = sys.version_info
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    except Exception as e:
        print(f"❌ Python check failed: {e}")
        return False
    
    # Check Git
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git not found. Please install Git first.")
        return False
    
    # Check Hugging Face CLI
    try:
        result = subprocess.run(["huggingface-cli", "--version"], capture_output=True, text=True, check=True)
        print(f"✅ Hugging Face CLI installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Hugging Face CLI not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            print("✅ Hugging Face CLI installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Hugging Face CLI: {e}")
            return False
    
    return True

def validate_model_files():
    """Validate that all required model files are present"""
    print_step(2, "Validating Model Files", "Checking that all required files are present")
    
    required_files = [
        "f1_prediction_model.joblib",
        "f1_scaler.joblib", 
        "feature_columns.csv",
        "feature_importance.csv",
        "README.md",
        "requirements.txt",
        "inference.py",
        "pipeline.py",
        "app.py",
        "config.json",
        "LICENSE"
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    
    print(f"\n✅ All {len(required_files)} required files are present")
    return True

def get_user_input():
    """Get user input for deployment"""
    print_step(3, "User Configuration", "Setting up deployment parameters")
    
    # Get username
    while True:
        username = input("Enter your Hugging Face username: ").strip()
        if username:
            break
        print("❌ Username is required")
    
    # Get repository name
    repo_name = input(f"Enter repository name (default: f1-race-predictor): ").strip()
    if not repo_name:
        repo_name = "f1-race-predictor"
    
    # Get visibility
    while True:
        visibility = input("Repository visibility (public/private, default: public): ").strip().lower()
        if not visibility:
            visibility = "public"
        if visibility in ["public", "private"]:
            break
        print("❌ Visibility must be 'public' or 'private'")
    
    # Get license
    while True:
        license_choice = input("License (MIT/Apache-2.0, default: MIT): ").strip().upper()
        if not license_choice:
            license_choice = "MIT"
        if license_choice in ["MIT", "APACHE-2.0"]:
            break
        print("❌ License must be 'MIT' or 'Apache-2.0'")
    
    return {
        'username': username,
        'repo_name': repo_name,
        'full_repo_name': f"{username}/{repo_name}",
        'visibility': visibility,
        'license': license_choice
    }

def login_to_huggingface():
    """Login to Hugging Face"""
    print_step(4, "Hugging Face Login", "Authenticating with Hugging Face")
    
    print("Please login to Hugging Face when prompted...")
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
        print("✅ Successfully logged in to Hugging Face")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to login: {e}")
        return False

def create_repository(config):
    """Create the repository on Hugging Face"""
    print_step(5, "Creating Repository", f"Creating {config['full_repo_name']} on Hugging Face")
    
    try:
        # Create repository
        cmd = [
            "huggingface-cli", "repo", "create", 
            config['full_repo_name'],
            "--type", "model",
            "--license", config['license']
        ]
        
        if config['visibility'] == 'private':
            cmd.append("--private")
        
        subprocess.run(cmd, check=True)
        print(f"✅ Repository {config['full_repo_name']} created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create repository: {e}")
        return False

def setup_git_repository(config):
    """Setup git repository and prepare for push"""
    print_step(6, "Setting Up Git Repository", "Initializing git and preparing files")
    
    try:
        # Initialize git
        subprocess.run(["git", "init"], check=True)
        print("✅ Git repository initialized")
        
        # Setup Git LFS
        subprocess.run(["git", "lfs", "install"], check=True)
        print("✅ Git LFS installed")
        
        # Track large files
        subprocess.run(["git", "lfs", "track", "*.joblib"], check=True)
        subprocess.run(["git", "lfs", "track", "*.pkl"], check=True)
        subprocess.run(["git", "lfs", "track", "*.h5"], check=True)
        print("✅ Git LFS tracking configured")
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print("✅ All files added to git")
        
        # Initial commit
        subprocess.run(["git", "commit", "-m", "Initial F1 2025 Race Predictor upload"], check=True)
        print("✅ Initial commit created")
        
        # Add remote
        remote_url = f"https://huggingface.co/{config['full_repo_name']}"
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        print(f"✅ Remote origin added: {remote_url}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git setup failed: {e}")
        return False

def push_to_huggingface(config):
    """Push the repository to Hugging Face"""
    print_step(7, "Pushing to Hugging Face", f"Uploading model to {config['full_repo_name']}")
    
    try:
        # Push to main branch
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"✅ Successfully pushed to {config['full_repo_name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push to Hugging Face: {e}")
        return False

def verify_deployment(config):
    """Verify the deployment was successful"""
    print_step(8, "Verifying Deployment", "Checking that the model is accessible")
    
    repo_url = f"https://huggingface.co/{config['full_repo_name']}"
    print(f"🔗 Repository URL: {repo_url}")
    
    print("\n📋 Next steps to verify:")
    print("1. Visit the repository URL above")
    print("2. Check that all files are present")
    print("3. Test the model using the provided examples")
    print("4. Try the Gradio app if available")
    
    return True

def create_usage_examples(config):
    """Create usage examples for the deployed model"""
    print_step(9, "Creating Usage Examples", "Generating code examples for users")
    
    repo_name = config['full_repo_name']
    
    examples = f"""
# 🏎️ F1 2025 Race Predictor - Usage Examples

## Quick Start

```python
from huggingface_hub import hf_hub_download
import joblib
from pipeline import F1PredictionPipeline

# Download model files
model_path = hf_hub_download(repo_id="{repo_name}", filename="f1_prediction_model.joblib")
scaler_path = hf_hub_download(repo_id="{repo_name}", filename="f1_scaler.joblib")

# Initialize pipeline
pipeline = F1PredictionPipeline(model_path, scaler_path)

# Make prediction
prediction = pipeline({{
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
}})

print(f"Win probability: {{prediction['win_probability']:.2%}}")
print(f"Podium probability: {{prediction['podium_probability']:.2%}}")
```

## Using the Inference API

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/{repo_name}"
headers = {{"Authorization": "Bearer YOUR_HF_TOKEN"}}

data = {{
    "inputs": {{
        "driver": "Lando Norris",
        "team": "McLaren", 
        "track_name": "British Grand Prix",
        "qualifying_position": 2,
        "recent_form_score": 0.88
    }}
}}

response = requests.post(API_URL, headers=headers, json=data)
result = response.json()
print(result)
```

## Running the Gradio App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
```

Visit the app at: http://localhost:7860
"""
    
    # Save examples to file
    with open("USAGE_EXAMPLES.md", "w") as f:
        f.write(examples)
    
    print("✅ Usage examples created in USAGE_EXAMPLES.md")
    return True

def main():
    """Main deployment function"""
    print("🏎️ F1 2025 Race Predictor - Hugging Face Deployment")
    print("=" * 60)
    print("This script will deploy your F1 prediction model to Hugging Face Hub")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites check failed. Please fix the issues above.")
        return False
    
    # Validate model files
    if not validate_model_files():
        print("❌ Model files validation failed. Please ensure all required files are present.")
        return False
    
    # Get user input
    config = get_user_input()
    
    # Login to Hugging Face
    if not login_to_huggingface():
        print("❌ Login failed. Please try again.")
        return False
    
    # Create repository
    if not create_repository(config):
        print("❌ Repository creation failed. Please try again.")
        return False
    
    # Setup git repository
    if not setup_git_repository(config):
        print("❌ Git setup failed. Please try again.")
        return False
    
    # Push to Hugging Face
    if not push_to_huggingface(config):
        print("❌ Push failed. Please try again.")
        return False
    
    # Verify deployment
    verify_deployment(config)
    
    # Create usage examples
    create_usage_examples(config)
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    repo_url = f"https://huggingface.co/{config['full_repo_name']}"
    print(f"\n✅ Your model is now live at: {repo_url}")
    
    print("\n📋 What's next:")
    print("1. Visit your repository to see the model")
    print("2. Test the model using the provided examples")
    print("3. Share your model with the community")
    print("4. Monitor usage and gather feedback")
    print("5. Update the model with new data as needed")
    
    print(f"\n🔗 Quick links:")
    print(f"   Repository: {repo_url}")
    print(f"   Usage Examples: USAGE_EXAMPLES.md")
    print(f"   Local Testing: python test_model.py")
    print(f"   Web App: python app.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 Deployment completed! Your F1 prediction model is now on Hugging Face Hub!")
        else:
            print("\n❌ Deployment failed. Please check the errors above and try again.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
