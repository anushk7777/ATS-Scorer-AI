#!/usr/bin/env python3
"""
ATS Resume Analyzer - Enhanced ML Model Training Script

This script trains an improved salary prediction model using semantically extracted
features from the Gemini API. It combines traditional resume parsing with advanced
NLP techniques for better accuracy.

Features:
- Semantic feature extraction using Gemini AI
- Advanced feature engineering
- Model training with cross-validation
- Performance evaluation and comparison
- Model persistence and versioning

Usage:
    python train_enhanced_model.py [--retrain] [--evaluate]

Author: AI Assistant
Date: 2024
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom imports
from gemini_semantic_extractor import GeminiSemanticExtractor
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    "logs/training.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


class EnhancedModelTrainer:
    """
    Enhanced ML model trainer that uses semantic features from Gemini API
    to improve salary prediction accuracy.
    """
    
    def __init__(self, gemini_extractor: GeminiSemanticExtractor = None):
        """
        Initialize the enhanced model trainer.
        
        Args:
            gemini_extractor: Instance of GeminiSemanticExtractor for semantic analysis
        """
        self.gemini_extractor = gemini_extractor or GeminiSemanticExtractor()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.model_metadata = {}
        
        # Create necessary directories
        Path("models").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("Enhanced Model Trainer initialized")
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic training data for demonstration purposes.
        In production, this would be replaced with real resume and salary data.
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            DataFrame with synthetic resume and salary data
        """
        logger.info(f"Generating {num_samples} synthetic training samples")
        
        # Define skill categories and their salary impact
        tech_skills = {
            'python': {'base_impact': 15000, 'seniority_multiplier': 1.3},
            'javascript': {'base_impact': 12000, 'seniority_multiplier': 1.2},
            'java': {'base_impact': 18000, 'seniority_multiplier': 1.4},
            'react': {'base_impact': 14000, 'seniority_multiplier': 1.25},
            'aws': {'base_impact': 20000, 'seniority_multiplier': 1.5},
            'docker': {'base_impact': 10000, 'seniority_multiplier': 1.2},
            'kubernetes': {'base_impact': 22000, 'seniority_multiplier': 1.6},
            'machine learning': {'base_impact': 25000, 'seniority_multiplier': 1.7},
            'data science': {'base_impact': 23000, 'seniority_multiplier': 1.6},
            'sql': {'base_impact': 8000, 'seniority_multiplier': 1.1}
        }
        
        job_titles = {
            'Software Engineer': {'base_salary': 80000, 'variance': 20000},
            'Senior Software Engineer': {'base_salary': 120000, 'variance': 30000},
            'Data Scientist': {'base_salary': 110000, 'variance': 25000},
            'DevOps Engineer': {'base_salary': 100000, 'variance': 25000},
            'Frontend Developer': {'base_salary': 75000, 'variance': 18000},
            'Backend Developer': {'base_salary': 85000, 'variance': 20000},
            'Full Stack Developer': {'base_salary': 90000, 'variance': 22000},
            'Machine Learning Engineer': {'base_salary': 130000, 'variance': 35000}
        }
        
        locations = {
            'San Francisco': 1.4,
            'New York': 1.3,
            'Seattle': 1.25,
            'Austin': 1.1,
            'Denver': 1.0,
            'Remote': 1.15
        }
        
        data = []
        
        for i in range(num_samples):
            # Random job title and base salary
            job_title = np.random.choice(list(job_titles.keys()))
            base_salary = job_titles[job_title]['base_salary']
            salary_variance = job_titles[job_title]['variance']
            
            # Random experience (0-15 years)
            years_experience = np.random.randint(0, 16)
            
            # Random location
            location = np.random.choice(list(locations.keys()))
            location_multiplier = locations[location]
            
            # Random skills (2-8 skills per person)
            num_skills = np.random.randint(2, 9)
            selected_skills = np.random.choice(list(tech_skills.keys()), num_skills, replace=False)
            
            # Calculate salary based on skills and experience
            skill_bonus = 0
            for skill in selected_skills:
                skill_data = tech_skills[skill]
                base_impact = skill_data['base_impact']
                seniority_multiplier = skill_data['seniority_multiplier']
                
                # Apply seniority multiplier based on experience
                experience_factor = min(years_experience / 10, 1.0)
                skill_bonus += base_impact * (1 + experience_factor * (seniority_multiplier - 1))
            
            # Experience bonus (5% per year, capped at 100%)
            experience_bonus = min(years_experience * 0.05, 1.0) * base_salary
            
            # Calculate final salary
            final_salary = (base_salary + skill_bonus + experience_bonus) * location_multiplier
            
            # Add some random variance
            final_salary += np.random.normal(0, salary_variance * 0.3)
            final_salary = max(final_salary, 30000)  # Minimum salary floor
            
            # Create resume text
            resume_text = self._generate_resume_text(
                job_title, selected_skills, years_experience, location
            )
            
            data.append({
                'resume_text': resume_text,
                'job_title': job_title,
                'location': location,
                'years_experience': years_experience,
                'skills': ', '.join(selected_skills),
                'num_skills': len(selected_skills),
                'salary': round(final_salary)
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic samples")
        return df
    
    def _generate_resume_text(self, job_title: str, skills: List[str], 
                             years_exp: int, location: str) -> str:
        """
        Generate realistic resume text for synthetic data.
        
        Args:
            job_title: Job title for the resume
            skills: List of technical skills
            years_exp: Years of experience
            location: Work location
            
        Returns:
            Generated resume text
        """
        experience_level = "Senior" if years_exp >= 5 else "Junior" if years_exp >= 2 else "Entry-level"
        
        resume_template = f"""
        {experience_level} {job_title}
        Location: {location}
        
        EXPERIENCE:
        {years_exp} years of professional experience in software development.
        
        TECHNICAL SKILLS:
        {', '.join(skills)}
        
        PROFESSIONAL SUMMARY:
        Experienced {job_title.lower()} with {years_exp} years of hands-on experience 
        in developing scalable applications. Proficient in {', '.join(skills[:3])} 
        and passionate about delivering high-quality software solutions.
        
        WORK EXPERIENCE:
        {job_title} - Tech Company ({years_exp//2 + 1} years)
        - Developed and maintained applications using {skills[0] if skills else 'various technologies'}
        - Collaborated with cross-functional teams to deliver projects on time
        - Implemented best practices for code quality and performance optimization
        """
        
        return resume_template.strip()
    
    def extract_semantic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract semantic features from resume text using Gemini API.
        
        Args:
            df: DataFrame with resume_text column
            
        Returns:
            DataFrame with additional semantic features
        """
        logger.info("Extracting semantic features using Gemini API")
        
        semantic_features = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing resume {idx + 1}/{len(df)}")
            
            try:
                # Extract skills using Gemini
                skills = self.gemini_extractor.extract_skills(row['resume_text'])
                experience = self.gemini_extractor.extract_experience(row['resume_text'])
                
                features = {
                    'semantic_tech_skills_count': len(skills.technical_skills),
                    'semantic_soft_skills_count': len(skills.soft_skills),
                    'semantic_certifications_count': len(skills.certifications),
                    'semantic_years_experience': experience.years_experience,
                    'semantic_experience_level': experience.experience_level,
                    'semantic_specializations': ', '.join(experience.specializations),
                    'semantic_skill_confidence': skills.confidence_score,
                    'semantic_experience_confidence': experience.confidence_score
                }
                
                semantic_features.append(features)
                
            except Exception as e:
                logger.warning(f"Failed to extract semantic features for resume {idx}: {e}")
                # Fallback to default values
                semantic_features.append({
                    'semantic_tech_skills_count': 0,
                    'semantic_soft_skills_count': 0,
                    'semantic_certifications_count': 0,
                    'semantic_years_experience': row.get('years_experience', 0),
                    'semantic_experience_level': 'entry',
                    'semantic_specializations': '',
                    'semantic_skill_confidence': 0.5,
                    'semantic_experience_confidence': 0.5
                })
        
        # Add semantic features to dataframe
        semantic_df = pd.DataFrame(semantic_features)
        result_df = pd.concat([df, semantic_df], axis=1)
        
        logger.info("Semantic feature extraction completed")
        return result_df
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features for machine learning model.
        
        Args:
            df: DataFrame with raw and semantic features
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        logger.info("Engineering features for ML model")
        
        # Encode categorical variables
        le_job_title = LabelEncoder()
        le_location = LabelEncoder()
        le_exp_level = LabelEncoder()
        
        df['job_title_encoded'] = le_job_title.fit_transform(df['job_title'])
        df['location_encoded'] = le_location.fit_transform(df['location'])
        df['semantic_experience_level_encoded'] = le_exp_level.fit_transform(
            df['semantic_experience_level']
        )
        
        # Store encoders for later use
        self.encoders['job_title'] = le_job_title
        self.encoders['location'] = le_location
        self.encoders['experience_level'] = le_exp_level
        
        # Create TF-IDF features for skills
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        skills_tfidf = tfidf.fit_transform(df['skills'].fillna('')).toarray()
        self.encoders['skills_tfidf'] = tfidf
        
        # Select numerical features
        numerical_features = [
            'years_experience',
            'num_skills',
            'semantic_tech_skills_count',
            'semantic_soft_skills_count',
            'semantic_certifications_count',
            'semantic_years_experience',
            'semantic_skill_confidence',
            'semantic_experience_confidence',
            'job_title_encoded',
            'location_encoded',
            'semantic_experience_level_encoded'
        ]
        
        # Combine all features
        X_numerical = df[numerical_features].values
        X_combined = np.hstack([X_numerical, skills_tfidf])
        
        # Store feature names
        self.feature_names = numerical_features + [f'skill_tfidf_{i}' for i in range(skills_tfidf.shape[1])]
        
        # Target variable
        y = df['salary'].values
        
        logger.info(f"Feature engineering completed. Shape: {X_combined.shape}")
        return X_combined, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple ML models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with training results and best model
        """
        logger.info("Training multiple ML models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['feature_scaler'] = scaler
        
        # Define models to train
        models_to_train = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge_regression': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}")
            
            # Train model
            if model_name in ['ridge_regression', 'linear_regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if model_name in ['ridge_regression', 'linear_regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[model_name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # Select best model based on R² score
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
        
        # Store best model
        self.models['salary_predictor'] = best_model
        self.model_metadata = {
            'best_model': best_model_name,
            'training_date': datetime.now().isoformat(),
            'feature_count': X.shape[1],
            'training_samples': X.shape[0],
            'performance': results[best_model_name]
        }
        
        return results
    
    def save_models(self, model_dir: str = "models") -> None:
        """
        Save trained models and metadata to disk.
        
        Args:
            model_dir: Directory to save models
        """
        logger.info(f"Saving models to {model_dir}")
        
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        # Save main model
        joblib.dump(self.models['salary_predictor'], model_path / 'enhanced_salary_model.pkl')
        
        # Save scalers and encoders
        joblib.dump(self.scalers, model_path / 'scalers.pkl')
        joblib.dump(self.encoders, model_path / 'encoders.pkl')
        
        # Save feature names
        joblib.dump(self.feature_names, model_path / 'feature_names.pkl')
        
        # Save metadata
        with open(model_path / 'model_metadata.json', 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        logger.info("Models saved successfully")
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        if 'salary_predictor' not in self.models:
            raise ValueError("No trained model found. Please train a model first.")
        
        model = self.models['salary_predictor']
        
        # Use appropriate features based on model type
        if isinstance(model, (Ridge, LinearRegression)):
            X_processed = self.scalers['feature_scaler'].transform(X)
        else:
            X_processed = X
        
        y_pred = model.predict(X_processed)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        return metrics


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train enhanced salary prediction model')
    parser.add_argument('--retrain', action='store_true', help='Force retrain even if model exists')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing model')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples')
    
    args = parser.parse_args()
    
    # Check if model already exists
    model_path = Path("models/enhanced_salary_model.pkl")
    if model_path.exists() and not args.retrain:
        logger.info("Enhanced model already exists. Use --retrain to force retraining.")
        if args.evaluate:
            logger.info("Evaluating existing model...")
            # Load and evaluate existing model
            return
        return
    
    try:
        # Initialize trainer
        logger.info("Initializing Enhanced Model Trainer")
        trainer = EnhancedModelTrainer()
        
        # Generate or load training data
        logger.info("Preparing training data")
        df = trainer.generate_synthetic_data(args.samples)
        
        # Extract semantic features (this would use real Gemini API in production)
        logger.info("Extracting semantic features")
        # For demo purposes, we'll simulate semantic features
        # In production, uncomment the next line:
        # df = trainer.extract_semantic_features(df)
        
        # For now, add mock semantic features
        df['semantic_tech_skills_count'] = df['num_skills'] * 0.8
        df['semantic_soft_skills_count'] = np.random.randint(2, 8, len(df))
        df['semantic_certifications_count'] = np.random.randint(0, 4, len(df))
        df['semantic_years_experience'] = df['years_experience']
        df['semantic_experience_level'] = df['years_experience'].apply(
            lambda x: 'senior' if x >= 5 else 'mid' if x >= 2 else 'entry'
        )
        df['semantic_skill_confidence'] = np.random.uniform(0.7, 0.95, len(df))
        df['semantic_experience_confidence'] = np.random.uniform(0.8, 0.98, len(df))
        
        # Engineer features
        X, y = trainer.engineer_features(df)
        
        # Train models
        results = trainer.train_models(X, y)
        
        # Save models
        trainer.save_models()
        
        # Print results
        print("\n" + "="*60)
        print("ENHANCED MODEL TRAINING RESULTS")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  RMSE: ${metrics['rmse']:,.2f}")
            print(f"  MAE: ${metrics['mae']:,.2f}")
            print(f"  CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        best_model = trainer.model_metadata['best_model']
        best_r2 = trainer.model_metadata['performance']['r2']
        print(f"\nBEST MODEL: {best_model.upper()} (R² = {best_r2:.4f})")
        print(f"Model saved to: models/enhanced_salary_model.pkl")
        
        logger.info("Enhanced model training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()