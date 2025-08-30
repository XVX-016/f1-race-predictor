#!/usr/bin/env python3
"""
Deploy F1 2025 Race Predictor to Hugging Face Hub
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_huggingface_cli():
    """Check if huggingface-cli is installed"""
    try:
        subprocess.run(["huggingface-cli", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_huggingface_cli():
    """Install huggingface-cli if not present"""
    print("Installing huggingface-cli...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        print("✅ huggingface-cli installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install huggingface-cli: {e}")
        return False

def login_to_huggingface():
    """Login to Hugging Face"""
    print("Please login to Hugging Face...")
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
        print("✅ Successfully logged in to Hugging Face")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to login: {e}")
        return False

def create_repo(repo_name):
    """Create a new repository on Hugging Face"""
    print(f"Creating repository: {repo_name}")
    try:
        subprocess.run(["huggingface-cli", "repo", "create", repo_name], check=True)
        print(f"✅ Repository {repo_name} created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create repository: {e}")
        return False

def initialize_git_repo():
    """Initialize git repository and add remote"""
    print("Initializing git repository...")
    try:
        # Initialize git
        subprocess.run(["git", "init"], check=True)
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        
        # Initial commit
        subprocess.run(["git", "commit", "-m", "Initial F1 predictor upload"], check=True)
        
        print("✅ Git repository initialized successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize git repository: {e}")
        return False

def setup_git_lfs():
    """Setup Git LFS for large files"""
    print("Setting up Git LFS...")
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
        
        # Track large files
        subprocess.run(["git", "lfs", "track", "*.joblib"], check=True)
        subprocess.run(["git", "lfs", "track", "*.pkl"], check=True)
        subprocess.run(["git", "lfs", "track", "*.h5"], check=True)
        
        # Add .gitattributes
        subprocess.run(["git", "add", ".gitattributes"], check=True)
        subprocess.run(["git", "commit", "-m", "Add Git LFS tracking"], check=True)
        
        print("✅ Git LFS setup completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup Git LFS: {e}")
        return False

def push_to_huggingface(repo_name):
    """Push the repository to Hugging Face"""
    print(f"Pushing to Hugging Face: {repo_name}")
    try:
        # Add remote
        remote_url = f"https://huggingface.co/{repo_name}"
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        
        # Push to main branch
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        print(f"✅ Successfully pushed to {remote_url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to push to Hugging Face: {e}")
        return False

def validate_files():
    """Validate that all required files are present"""
    print("Validating required files...")
    
    required_files = [
        "f1_prediction_model.joblib",
        "f1_scaler.joblib",
        "feature_columns.csv",
        "feature_importance.csv",
        "README.md",
        "requirements.txt",
        "inference.py",
        "config.json",
        "LICENSE"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    """Main deployment function"""
    print("🚀 F1 2025 Race Predictor - Hugging Face Deployment")
    print("=" * 60)
    
    # Get repository name
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ Username is required")
        return
    
    repo_name = f"{username}/f1-race-predictor"
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first.")
        return
    
    if not check_huggingface_cli():
        if not install_huggingface_cli():
            return
    
    # Validate files
    if not validate_files():
        return
    
    # Login to Hugging Face
    if not login_to_huggingface():
        return
    
    # Create repository
    if not create_repo(repo_name):
        return
    
    # Setup git repository
    if not initialize_git_repo():
        return
    
    # Setup Git LFS
    if not setup_git_lfs():
        return
    
    # Push to Hugging Face
    if not push_to_huggingface(repo_name):
        return
    
    print("\n" + "=" * 60)
    print("🎉 Deployment completed successfully!")
    print(f"\nYour model is now available at:")
    print(f"https://huggingface.co/{repo_name}")
    
    print("\n📋 Next steps:")
    print("1. Visit your repository on Hugging Face Hub")
    print("2. Add a model card with detailed information")
    print("3. Test the model using the provided examples")
    print("4. Share your model with the community")
    
    print("\n🔗 Usage examples:")
    print(f"```python")
    print(f"from huggingface_hub import hf_hub_download")
    print(f"import joblib")
    print(f"")
    print(f"# Download model")
    print(f"model_path = hf_hub_download(repo_id='{repo_name}', filename='f1_prediction_model.joblib')")
    print(f"model = joblib.load(model_path)")
    print(f"```")

if __name__ == "__main__":
    main()
