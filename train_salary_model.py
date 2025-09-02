#!/usr/bin/env python3
"""
Salary Prediction Model Training Script

This script trains a machine learning model to predict salaries based on resume features.
It uses the synthetic employee data generated earlier and exports the trained model
for client-side inference in the ATS application.

Features:
- Data preprocessing and feature engineering
- Multiple model training and comparison
- Model evaluation and validation
- Export trained model for JavaScript inference
- Generate feature importance analysis

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictionTrainer:
    """
    A comprehensive salary prediction model trainer that handles data preprocessing,
    model training, evaluation, and export for client-side inference.
    """
    
    def __init__(self, data_path='synthetic_employee_data.csv'):
        """
        Initialize the trainer with data path.
        
        Args:
            data_path (str): Path to the synthetic employee data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_names = []
        self.model_performance = {}
        
    def load_and_preprocess_data(self):
        """
        Load the synthetic data and perform preprocessing including:
        - Feature encoding for categorical variables
        - Feature scaling for numerical variables
        - Feature selection and engineering
        """
        print("Loading synthetic employee data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} records with {len(self.data.columns)} features")
        
        # Define features to use for training
        categorical_features = [
            'experience_group', 'role', 'location', 'education_level',
            'industry_experience', 'remote_work_preference', 'career_level'
        ]
        
        numerical_features = [
            'years_of_experience', 'ats_scores', 'market_rate', 'performance_rating',
            'skill_proficiency', 'leadership_score', 'technical_skills_count',
            'certifications_count', 'projects_completed', 'team_size_managed',
            'budget_managed', 'revenue_generated', 'process_improvements',
            'training_hours', 'mentorship_given', 'cross_functional_experience'
        ]
        
        # Prepare feature matrix
        X = pd.DataFrame()
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in self.data.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(self.data[feature].astype(str))
                self.label_encoders[feature] = le
                print(f"Encoded {feature}: {len(le.classes_)} unique values")
        
        # Add numerical features
        for feature in numerical_features:
            if feature in self.data.columns:
                X[feature] = self.data[feature]
        
        # Feature engineering - create interaction features
        if 'years_of_experience' in X.columns and 'skill_proficiency' in X.columns:
            X['experience_skill_interaction'] = X['years_of_experience'] * X['skill_proficiency']
        
        if 'ats_scores' in X.columns and 'performance_rating' in X.columns:
            X['ats_performance_interaction'] = X['ats_scores'] * X['performance_rating']
        
        if 'leadership_score' in X.columns and 'team_size_managed' in X.columns:
            X['leadership_impact'] = X['leadership_score'] * np.log1p(X['team_size_managed'])
        
        # Target variable
        y = self.data['current_salary']
        
        # Store feature names
        self.feature_names = list(X.columns)
        print(f"Final feature set: {len(self.feature_names)} features")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=self.data['experience_group']
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
    def train_models(self):
        """
        Train multiple machine learning models and compare their performance.
        Models include Random Forest, Gradient Boosting, and Linear Regression.
        """
        print("\nTraining multiple models...")
        
        # Define models to train
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Linear Regression': LinearRegression()
        }
        
        best_score = -np.inf
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Linear Regression, original for tree-based models
            if name == 'Linear Regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_use)
            y_pred_test = model.predict(X_test_use)
            
            # Calculate metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_use, self.y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store performance metrics
            self.model_performance[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: ${test_mae:,.0f}")
            print(f"  Test RMSE: ${test_rmse:,.0f}")
            print(f"  CV R² (mean ± std): {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Select best model based on cross-validation score
            if cv_mean > best_score:
                best_score = cv_mean
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} (CV R²: {best_score:.4f})")
        
    def analyze_feature_importance(self):
        """
        Analyze and display feature importance for the best model.
        """
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\nFeature Importance Analysis ({self.best_model_name}):")
            
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
            
            return feature_importance
        else:
            print(f"\nFeature importance not available for {self.best_model_name}")
            return None
    
    def export_model_for_inference(self):
        """
        Export the trained model and preprocessing components for client-side inference.
        Creates both a joblib file and a JSON configuration for JavaScript usage.
        """
        print("\nExporting model for inference...")
        
        # Save the complete model pipeline
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'performance': self.model_performance[self.best_model_name]
        }
        
        # Save as joblib for Python usage
        joblib.dump(model_data, 'salary_prediction_model.joblib')
        print("Saved complete model pipeline: salary_prediction_model.joblib")
        
        # Create JavaScript-compatible configuration
        js_config = {
            'model_type': self.best_model_name,
            'feature_names': self.feature_names,
            'performance_metrics': {
                'test_r2': float(self.model_performance[self.best_model_name]['test_r2']),
                'test_mae': float(self.model_performance[self.best_model_name]['test_mae']),
                'test_rmse': float(self.model_performance[self.best_model_name]['test_rmse'])
            },
            'label_encoders': {},
            'scaler_params': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }
        
        # Export label encoder mappings
        for feature, encoder in self.label_encoders.items():
            js_config['label_encoders'][feature] = {
                'classes': encoder.classes_.tolist()
            }
        
        # Export model parameters for tree-based models
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            # For tree-based models, we'll need to implement a simplified version
            # or use a library like sklearn-porter for JavaScript export
            js_config['model_params'] = {
                'note': 'Tree-based model - requires server-side prediction or ONNX export'
            }
        
        # Save JavaScript configuration
        with open('salary_model_config.json', 'w') as f:
            json.dump(js_config, f, indent=2)
        print("Saved JavaScript configuration: salary_model_config.json")
        
        # Create a simple prediction function template
        self.create_prediction_template()
        
    def create_prediction_template(self):
        """
        Create a Python template for making predictions with the trained model.
        """
        template = f'''
#!/usr/bin/env python3
"""
Salary Prediction Inference Template

This script demonstrates how to use the trained salary prediction model
to make predictions on new data.
"""

import joblib
import numpy as np
import pandas as pd

def load_model():
    """Load the trained salary prediction model."""
    return joblib.load('salary_prediction_model.joblib')

def predict_salary(model_data, resume_features):
    """
    Predict salary based on resume features.
    
    Args:
        model_data: Loaded model data from joblib
        resume_features: Dictionary of resume features
    
    Returns:
        Predicted salary as float
    """
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    
    # Create feature vector
    features = []
    for feature_name in feature_names:
        if feature_name in resume_features:
            value = resume_features[feature_name]
            
            # Apply label encoding if needed
            if feature_name in label_encoders:
                try:
                    value = label_encoders[feature_name].transform([str(value)])[0]
                except ValueError:
                    # Handle unknown categories
                    value = 0
            
            features.append(value)
        else:
            # Default value for missing features
            features.append(0)
    
    # Convert to numpy array and reshape
    X = np.array(features).reshape(1, -1)
    
    # Apply scaling if needed (for Linear Regression)
    if model_data['model_name'] == 'Linear Regression':
        X = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction

# Example usage
if __name__ == "__main__":
    # Load the model
    model_data = load_model()
    
    # Example resume features
    example_features = {{
        'years_of_experience': 5,
        'ats_scores': 85,
        'skill_proficiency': 8,
        'performance_rating': 4,
        'experience_group': 'Mid-Level',
        'role': 'Software Engineer',
        'location': 'San Francisco',
        'education_level': 'Bachelor',
        'leadership_score': 6,
        'technical_skills_count': 12,
        'certifications_count': 3
    }}
    
    # Predict salary
    predicted_salary = predict_salary(model_data, example_features)
    print(f"Predicted salary: ${{predicted_salary:,.0f}}")
'''
        
        with open('predict_salary.py', 'w') as f:
            f.write(template)
        print("Created prediction template: predict_salary.py")
    
    def generate_training_report(self):
        """
        Generate a comprehensive training report with model performance and insights.
        """
        print("\nGenerating training report...")
        
        report = {
            'training_summary': {
                'dataset_size': len(self.data),
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'feature_count': len(self.feature_names),
                'best_model': self.best_model_name
            },
            'model_performance': self.model_performance,
            'feature_names': self.feature_names,
            'data_statistics': {
                'salary_mean': float(self.data['current_salary'].mean()),
                'salary_std': float(self.data['current_salary'].std()),
                'salary_min': float(self.data['current_salary'].min()),
                'salary_max': float(self.data['current_salary'].max())
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        for model_name, metrics in report['model_performance'].items():
            for key, value in metrics.items():
                if key != 'model' and isinstance(value, (np.integer, np.floating)):
                    report['model_performance'][model_name][key] = float(value)
            # Remove the model object for JSON serialization
            if 'model' in report['model_performance'][model_name]:
                del report['model_performance'][model_name]['model']
        
        # Save report
        with open('training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("Saved training report: training_report.json")
        
        return report

def main():
    """
    Main training pipeline that orchestrates the entire model training process.
    """
    print("=" * 60)
    print("SALARY PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SalaryPredictionTrainer()
    
    try:
        # Load and preprocess data
        trainer.load_and_preprocess_data()
        
        # Train models
        trainer.train_models()
        
        # Analyze feature importance
        trainer.analyze_feature_importance()
        
        # Export model for inference
        trainer.export_model_for_inference()
        
        # Generate training report
        trainer.generate_training_report()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - salary_prediction_model.joblib (Complete model pipeline)")
        print("  - salary_model_config.json (JavaScript configuration)")
        print("  - predict_salary.py (Prediction template)")
        print("  - training_report.json (Training report)")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()