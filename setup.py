#!/usr/bin/env python3
"""
Setup script for F1 2025 Race Predictor
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="f1-race-predictor",
    version="1.0.0",
    author="F1 Prediction Team",
    author_email="contact@f1predictor.com",
    description="Advanced Formula 1 race outcome prediction model using ensemble machine learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://huggingface.co/your-username/f1-race-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.joblib", "*.csv", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "f1-predictor=inference:main",
        ],
    },
    keywords="f1 formula1 racing prediction machine-learning scikit-learn",
    project_urls={
        "Bug Reports": "https://github.com/your-username/f1-race-predictor/issues",
        "Source": "https://github.com/your-username/f1-race-predictor",
        "Documentation": "https://huggingface.co/your-username/f1-race-predictor",
    },
)
