#!/usr/bin/env python3
"""
Diabetes Model Integration for Hygieia Application
Connects the main Hygieia application to the recreated diabetes classification model
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union

class DiabetesModelIntegration:
    """
    Integration class for diabetes prediction model in the Hygieia application
    
    This class provides a clean interface between the main Hygieia app and the
    diabetes classification model, handling all preprocessing, prediction, and
    result interpretation.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the diabetes model integration
        
        Args:
            model_path: Path to the model directory. If None, uses default path.
        """
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'models', 'Diabetes_Model')
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.is_loaded = False
        
        # Expected input features for UCI diabetes model
        self.input_features = [
            'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 
            'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 
            'Itching', 'Irritability', 'delayed healing', 'partial paresis', 
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        
        # Feature descriptions for UI display
        self.feature_descriptions = {
            'Age': 'Age (years)',
            'Gender': 'Gender (Male/Female)',
            'Polyuria': 'Excessive urination',
            'Polydipsia': 'Excessive thirst',
            'sudden weight loss': 'Sudden weight loss',
            'weakness': 'General weakness',
            'Polyphagia': 'Excessive hunger',
            'Genital thrush': 'Genital thrush',
            'visual blurring': 'Blurred vision',
            'Itching': 'Itching',
            'Irritability': 'Irritability',
            'delayed healing': 'Delayed wound healing',
            'partial paresis': 'Partial paralysis',
            'muscle stiffness': 'Muscle stiffness',
            'Alopecia': 'Hair loss',
            'Obesity': 'Obesity'
        }
        
        # Load model automatically
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the UCI diabetes model and associated components
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load the UCI diabetes model
            model_file_path = os.path.join(self.model_path, 'diab_model.joblib')
            
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model file not found: {model_file_path}")
            
            # Load model data
            model_data = joblib.load(model_file_path)
            
            # Extract components from the dictionary
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.label_encoders = model_data.get('label_encoders')
                self.feature_names = model_data.get('feature_names')
                print(f"✅ Loaded UCI diabetes model from {model_file_path}")
                print(f"✅ Model type: {model_data.get('model_type', 'Unknown')}")
                if self.feature_names:
                    print(f"✅ Features: {len(self.feature_names)} features loaded")
            else:
                self.model = model_data
                print(f"✅ Loaded diabetes model from {model_file_path}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading diabetes model: {e}")
            self.is_loaded = False
            return False
    
    def validate_input(self, data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate input data for UCI diabetes prediction
        
        Args:
            data: Dictionary with feature names as keys and values
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if all required features are present
        for feature in self.input_features:
            if feature not in data:
                errors.append(f"Missing required feature: {feature}")
        
        # Check for invalid values
        if 'Age' in data and (data['Age'] < 0 or data['Age'] > 120):
            errors.append("Age must be between 0 and 120 years")
            
        if 'Gender' in data and data['Gender'] not in ['Male', 'Female']:
            errors.append("Gender must be 'Male' or 'Female'")
        
        # Check Yes/No values for symptom features
        yes_no_features = [f for f in self.input_features if f not in ['Age', 'Gender']]
        for feature in yes_no_features:
            if feature in data and data[feature] not in ['Yes', 'No', 0, 1]:
                errors.append(f"{feature} must be 'Yes' or 'No'")
        
        return len(errors) == 0, errors
    
    def _create_features(self, data: Dict[str, float]) -> np.ndarray:
        """
        Create feature array from UCI diabetes input data
        
        Args:
            data: Raw input data dictionary
            
        Returns:
            numpy array with all features in correct order
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([data])
        
        # Apply label encoding to categorical features
        if self.label_encoders:
            for col in df.columns:
                if col in self.label_encoders and col != 'Age':  # Age is numeric
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Get features in the correct order
        if self.feature_names:
            # Reorder columns to match training order
            features = df[self.feature_names].values
        else:
            # Use default order if feature names not available
            features = df[self.input_features].values
        
        return features
    
    def predict_diabetes_risk(self, patient_data: Dict[str, Union[str, int, float]]) -> Dict[str, Union[int, float, str, List]]:
        """
        Predict diabetes risk for a patient using UCI symptom-based model
        
        Args:
            patient_data: Dictionary with patient data
                         Expected keys: Age, Gender, Polyuria, Polydipsia, sudden weight loss,
                                      weakness, Polyphagia, Genital thrush, visual blurring,
                                      Itching, Irritability, delayed healing, partial paresis,
                                      muscle stiffness, Alopecia, Obesity
        
        Returns:
            Dictionary containing:
                - prediction: 0 (No Diabetes) or 1 (Diabetes)
                - probability: Probability of having diabetes (0-1)
                - risk_level: 'Low', 'Medium', 'High'
                - confidence: Model confidence (0-1)
                - interpretation: Human-readable interpretation
                - risk_factors: List of identified risk factors
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded. Please check model files.',
                'prediction': None,
                'probability': None,
                'success': False
            }
        
        # Validate input
        is_valid, errors = self.validate_input(patient_data)
        if not is_valid:
            return {
                'error': f"Invalid input: {'; '.join(errors)}",
                'prediction': None,
                'probability': None,
                'success': False
            }
        
        try:
            # Create features
            features = self._create_features(patient_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get probability if available
            probability = 0.5  # Default
            if hasattr(self.model, 'predict_proba'):
                prob_array = self.model.predict_proba(features)[0]
                probability = prob_array[1]  # Probability of class 1 (diabetes)
            
            # Determine risk level and calculate proper confidence
            if probability < 0.3:
                risk_level = 'Low'
                # For low risk, confidence increases as probability decreases
                confidence = min(0.95, 0.7 + (0.3 - probability) * 0.8)
            elif probability < 0.7:
                risk_level = 'Medium'
                # For medium risk, confidence is based on distance from boundaries
                confidence = min(0.85, 0.6 + min(abs(probability - 0.3), abs(probability - 0.7)) * 2)
            else:
                risk_level = 'High'
                # For high risk, confidence increases as probability increases
                confidence = min(0.95, 0.7 + (probability - 0.7) * 0.8)
            
            # Generate interpretation
            if prediction == 0:
                interpretation = f"Low risk of diabetes based on symptoms (Probability: {probability:.1%})"
            else:
                interpretation = f"High risk of diabetes based on symptoms (Probability: {probability:.1%})"
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(patient_data)
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'confidence': float(confidence),
                'interpretation': interpretation,
                'risk_factors': risk_factors,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f"Prediction error: {str(e)}",
                'prediction': None,
                'probability': None,
                'success': False
            }
    
    def _identify_risk_factors(self, patient_data: Dict[str, Union[str, int, float]]) -> List[str]:
        """
        Identify potential risk factors based on UCI symptom data
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            List of identified risk factors
        """
        risk_factors = []
        
        # Map symptoms to human-readable descriptions
        symptom_descriptions = {
            'Polyuria': 'Frequent urination',
            'Polydipsia': 'Excessive thirst',
            'sudden weight loss': 'Sudden weight loss',
            'weakness': 'General weakness',
            'Polyphagia': 'Excessive hunger',
            'Genital thrush': 'Genital thrush infection',
            'visual blurring': 'Blurred vision',
            'Itching': 'Persistent itching',
            'Irritability': 'Irritability',
            'delayed healing': 'Delayed wound healing',
            'partial paresis': 'Partial paralysis',
            'muscle stiffness': 'Muscle stiffness',
            'Alopecia': 'Hair loss',
            'Obesity': 'Obesity'
        }
        
        # Check for present symptoms
        for symptom, description in symptom_descriptions.items():
            if patient_data.get(symptom) in ['Yes', 1]:
                risk_factors.append(description)
        
        # Age-related risk
        age = patient_data.get('Age', 0)
        if age > 45:
            risk_factors.append(f"Advanced age ({age} years)")
        
        return risk_factors

    def fallback_diabetes_prediction(self, input_data: Dict[str, Union[str, int, float]]) -> Dict[str, Union[int, float, str, List]]:
        """
        Fallback diabetes prediction using basic symptom counting.
        Used when the main model is unavailable.
        
        Args:
            input_data: Dictionary with patient data
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Count number of symptoms present
            symptom_features = [
                'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
                'Irritability', 'delayed healing', 'partial paresis',
                'muscle stiffness', 'Alopecia', 'Obesity'
            ]
            
            symptom_count = sum(1 for symptom in symptom_features 
                              if input_data.get(symptom) in ['Yes', 1])
            
            # Calculate risk based on symptom count
            total_symptoms = len(symptom_features)
            symptom_ratio = symptom_count / total_symptoms
            
            # Determine prediction
            if symptom_ratio > 0.5:  # More than 50% symptoms present
                prediction = 'High Risk'
                confidence = min(0.90, 0.70 + symptom_ratio * 0.3)
            elif symptom_ratio > 0.25:  # 25-50% symptoms present
                prediction = 'Moderate Risk'
                confidence = min(0.85, 0.65 + symptom_ratio * 0.3)
            else:  # Less than 25% symptoms
                prediction = 'Low Risk'
                confidence = min(0.90, 0.75 + (1 - symptom_ratio) * 0.2)
            
            return {
                'prediction': prediction,
                'confidence': round(float(confidence), 3),
                'symptom_count': symptom_count,
                'total_symptoms': total_symptoms,
                'model_type': 'fallback',
                'success': True
            }
            
        except Exception as e:
            return {
                'prediction': 'Analysis Error',
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }

# Convenience function for easy import
def create_diabetes_predictor(model_path: str = None) -> DiabetesModelIntegration:
    """
    Create a diabetes predictor instance
    
    Args:
        model_path: Path to model directory
        
    Returns:
        DiabetesModelIntegration instance
    """
    return DiabetesModelIntegration(model_path)

#------------------------------------------------------------------------------
# Example usage and testing
if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing Diabetes Model Integration")
    print("=" * 40)
    
    # Create predictor
    predictor = create_diabetes_predictor()
    
    # Test with sample data
    sample_patient = {
        'Age': 45,
        'Gender': 'Male',
        'Polyuria': 'Yes',
        'Polydipsia': 'Yes',
        'sudden weight loss': 'No',
        'weakness': 'Yes',
        'Polyphagia': 'No',
        'Genital thrush': 'No',
        'visual blurring': 'Yes',
        'Itching': 'No',
        'Irritability': 'No',
        'delayed healing': 'No',
        'partial paresis': 'No',
        'muscle stiffness': 'No',
        'Alopecia': 'No',
        'Obesity': 'Yes'
    }
    
    print("Sample Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    print("\nPrediction Result:")
    result = predictor.predict_diabetes_risk(sample_patient)
    
    if result.get('success', False):
        print(f"  Prediction: {'Diabetes Risk' if result['prediction'] == 1 else 'No Diabetes Risk'}")
        print(f"  Probability: {result['probability']:.1%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Interpretation: {result['interpretation']}")
        print(f"  Risk Factors: {', '.join(result['risk_factors']) if result['risk_factors'] else 'None identified'}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print("\nModel Information:")
    print(f"  Model loaded: {predictor.is_loaded}")
    print(f"  Features expected: {len(predictor.input_features)}")
    print(f"  Feature names: {', '.join(predictor.input_features[:5])}...")

# Start
# |
# v
# predict_diabetes_risk(patient_data)
# |
# +--> validate_input(patient_data)
# |      |
# |      +--> Check for missing required features
# |      +--> Check if feature values are in valid medical ranges
# |
# +--> _create_features(patient_data)
# |      |
# |      +--> Handle zero values by imputing medians
# |      |
# |      +--> Engineer new features (N1, N2, N0, etc.)
# |      |
# |      +--> Scale numerical features using the loaded scaler
# |
# +--> Predict using loaded model
# |      |
# |      +--> model.predict(features)          -> Get binary prediction (0 or 1)
# |      +--> model.predict_proba(features)    -> Get class probabilities
# |
# +--> Determine risk level and confidence
# |      |
# |      +--> Categorize probability into 'Low', 'Medium', 'High'
# |      +--> Calculate a confidence score based on the probability
# |
# +--> _identify_risk_factors(patient_data)
# |      |
# |      +--> Identify clinical risk factors based on rules (e.g., BMI > 30)
# |
# v
# Format and return final result dictionary
# |
# v
# End