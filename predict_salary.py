
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
    
    # Example resume features (based on our training data)
    example_features = {
        'experience_group': 'mid_level',  # freshers, junior, mid_level, senior
        'role': 'Backend Developer',      # One of the 10 roles in training data
        'location': 'Mumbai',             # One of the 8 Indian cities
        'education_level': 'Master',      # Bachelor, Master, PhD
        'years_of_experience': 3,         # Numerical value
        'market_rate': 18.5,             # Market rate in lakhs
        'performance_rating': 4.2,        # Performance rating (1-5)
        'leadership_score': 7.5,          # Leadership score
        'cross_functional_experience': 2   # Cross-functional experience
    }
    
    # Predict salary
    predicted_salary = predict_salary(model_data, example_features)
    print(f"Predicted salary: ₹{predicted_salary:.1f} lakhs per annum")
    print(f"Predicted salary: ₹{predicted_salary*100000:,.0f} per annum")
