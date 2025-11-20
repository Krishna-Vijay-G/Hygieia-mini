#!/usr/bin/env python3
"""
Post-processing calibration for dermatology model predictions.

Model v4.0 Performance: 95.9% peak accuracy, 93.9% mean production accuracy on 8,039 samples
This module addresses class imbalance bias by calibrating prediction probabilities
to reduce bias toward majority classes (especially 'nv' which comprises 66.9% of HAM10000 data).

Calibration Parameters (v4.0 optimized):
- Temperature: 1.08 (reduced from v3.0: 1.15)
- Prior adjustment: 0.15 (reduced from v3.0: 0.25)
"""

import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Class priors from HAM10000 dataset (calculated from metadata)
CLASS_PRIORS = {
    'nv': 0.669,     # Melanocytic nevi (most common - 66.9%)
    'mel': 0.111,    # Melanoma (11.1%)
    'bkl': 0.110,    # Benign keratosis (11.0%)
    'bcc': 0.051,    # Basal cell carcinoma (5.1%)
    'akiec': 0.033,  # Actinic keratosis (3.3%)
    'vasc': 0.014,   # Vascular lesions (1.4%)
    'df': 0.011      # Dermatofibroma (1.1%)
}

# Calibration parameters (optimized for v4.0 dermatology model)
CALIBRATION_CONFIG = {
    'temperature': 1.08,  # OPTIMIZED v4.0: Fine-tuned for 95.9% peak accuracy (reduced from v3.0: 1.15)
    'prior_adjustment_strength': 0.15,  # OPTIMIZED v4.0: Conservative to preserve 93.9% production accuracy (reduced from v3.0: 0.25)
    'min_confidence_threshold': 0.1,   # Minimum confidence to maintain
    'max_confidence_threshold': 0.95   # Maximum confidence to prevent overconfidence
}

def calibrate_prediction_probabilities(raw_probabilities: Dict[str, float],
                                     method: str = 'combined') -> Dict[str, float]:
    """
    Calibrate prediction probabilities to reduce class imbalance bias.

    Args:
        raw_probabilities: Raw prediction probabilities from model
        method: Calibration method ('prior', 'temperature', 'combined')

    Returns:
        Calibrated probabilities dictionary
    """
    try:
        # Convert to numpy array for easier manipulation
        classes = list(raw_probabilities.keys())
        probs = np.array([raw_probabilities[cls] for cls in classes])

        # Ensure probabilities sum to 1
        probs = probs / np.sum(probs)

        if method == 'prior':
            calibrated_probs = _apply_prior_adjustment(probs, classes)
        elif method == 'temperature':
            calibrated_probs = _apply_temperature_scaling(probs)
        elif method == 'combined':
            # Apply both prior adjustment and temperature scaling
            probs_temp = _apply_temperature_scaling(probs)
            calibrated_probs = _apply_prior_adjustment(probs_temp, classes)
        else:
            logger.warning(f"Unknown calibration method: {method}, using raw probabilities")
            calibrated_probs = probs

        # Apply confidence bounds
        calibrated_probs = _apply_confidence_bounds(calibrated_probs)

        # Ensure probabilities sum to 1 after calibration
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)

        # Convert back to dictionary
        calibrated_dict = {classes[i]: float(calibrated_probs[i]) for i in range(len(classes))}

        return calibrated_dict

    except Exception as e:
        logger.error(f"Error in probability calibration: {e}")
        return raw_probabilities

def _apply_prior_adjustment(probabilities: np.ndarray, classes: list) -> np.ndarray:
    """
    Adjust probabilities based on class priors to reduce majority class bias.

    This divides each probability by its class prior, then renormalizes.
    This reduces the advantage of majority classes while preserving relative rankings.
    """
    adjusted_probs = probabilities.copy()

    for i, cls in enumerate(classes):
        if cls in CLASS_PRIORS:
            prior = CLASS_PRIORS[cls]
            # Adjust probability by dividing by prior (inverse prior weighting)
            # Strength parameter controls how strongly we apply this adjustment
            strength = CALIBRATION_CONFIG['prior_adjustment_strength']
            adjusted_probs[i] = probabilities[i] * (1.0 / (prior ** strength))

    # Renormalize to ensure probabilities sum to 1
    adjusted_probs = adjusted_probs / np.sum(adjusted_probs)

    return adjusted_probs

def _apply_temperature_scaling(probabilities: np.ndarray) -> np.ndarray:
    """
    Apply temperature scaling to reduce overconfidence in predictions.

    Temperature scaling divides logits by a temperature parameter before softmax.
    For probabilities, we can achieve similar effect by raising to power of (1/temperature).
    """
    temperature = CALIBRATION_CONFIG['temperature']

    if temperature == 1.0:
        return probabilities

    # Apply temperature scaling: p_i^(1/T) / sum(p_j^(1/T))
    scaled_probs = probabilities ** (1.0 / temperature)
    scaled_probs = scaled_probs / np.sum(scaled_probs)

    return scaled_probs

def _apply_confidence_bounds(probabilities: np.ndarray) -> np.ndarray:
    """
    Apply minimum and maximum confidence bounds to prevent extreme values.
    """
    min_conf = CALIBRATION_CONFIG['min_confidence_threshold']
    max_conf = CALIBRATION_CONFIG['max_confidence_threshold']

    # Apply bounds
    bounded_probs = np.clip(probabilities, min_conf, max_conf)

    # Renormalize after bounding
    bounded_probs = bounded_probs / np.sum(bounded_probs)

    return bounded_probs

def calibrate_prediction_result(prediction_result: Dict[str, Any],
                              calibration_method: str = 'combined') -> Dict[str, Any]:
    
    """Apply calibration to a complete prediction result dictionary."""
    try:
        # Make a copy to avoid modifying the original
        calibrated_result = prediction_result.copy()

        # Check if we have probabilities to calibrate
        if 'probabilities' not in calibrated_result or not calibrated_result['probabilities']:
            logger.warning("No probabilities found in prediction result, skipping calibration")
            calibrated_result['calibration_applied'] = False
            return calibrated_result

        # Store original probabilities for comparison
        original_probabilities = calibrated_result['probabilities'].copy()

        # Apply calibration
        calibrated_probabilities = calibrate_prediction_probabilities(
            original_probabilities,
            method=calibration_method
        )

        # Update the result with calibrated probabilities
        calibrated_result['probabilities'] = calibrated_probabilities

        # Update confidence (maximum probability)
        calibrated_result['confidence'] = max(calibrated_probabilities.values())

        # Update prediction if it changed due to calibration
        original_prediction = calibrated_result.get('prediction', '')
        new_prediction = max(calibrated_probabilities, key=calibrated_probabilities.get)

        if new_prediction != original_prediction:
            logger.info(f"Prediction changed due to calibration: {original_prediction} -> {new_prediction}")
            calibrated_result['prediction'] = new_prediction

            # Update condition name and risk level if prediction changed
            from derm_model import CONDITION_NAMES, RISK_LEVELS
            calibrated_result['condition'] = CONDITION_NAMES.get(new_prediction, new_prediction)
            calibrated_result['condition_name'] = CONDITION_NAMES.get(new_prediction, new_prediction)
            calibrated_result['risk_level'] = RISK_LEVELS.get(new_prediction, 'Unknown')

        # Add calibration metadata
        calibrated_result['calibration_applied'] = True
        calibrated_result['calibration_method'] = calibration_method
        calibrated_result['original_probabilities'] = original_probabilities
        calibrated_result['calibration_info'] = {
            'temperature': CALIBRATION_CONFIG['temperature'],
            'prior_adjustment_strength': CALIBRATION_CONFIG['prior_adjustment_strength'],
            'class_priors': CLASS_PRIORS
        }

        logger.info(f"Applied {calibration_method} calibration: confidence {max(original_probabilities.values()):.3f} -> {calibrated_result['confidence']:.3f}")

        return calibrated_result

    except Exception as e:
        logger.error(f"Error applying calibration: {e}")
        calibrated_result['calibration_applied'] = False
        calibrated_result['calibration_error'] = str(e)
        return calibrated_result




'''the following funtions are utility functions for experimentation only - not used during prediction 
remove the comment to use the functions'''


'''<--- remove for experimentsation only - not used during prediction
def get_calibration_stats(original_probs: Dict[str, float],
                         calibrated_probs: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate statistics about the calibration effect.

    Args:
        original_probs: Original prediction probabilities
        calibrated_probs: Calibrated prediction probabilities

    Returns:
        Dictionary with calibration statistics
    """
    try:
        # Calculate changes in probabilities
        changes = {}
        for cls in original_probs.keys():
            if cls in calibrated_probs:
                changes[cls] = calibrated_probs[cls] - original_probs[cls]

        # Find classes with largest changes
        most_increased = max(changes, key=lambda x: changes[x]) if changes else None
        most_decreased = min(changes, key=lambda x: changes[x]) if changes else None

        # Calculate entropy change (measure of confidence distribution)
        orig_entropy = -sum(p * np.log(p + 1e-10) for p in original_probs.values())
        cal_entropy = -sum(p * np.log(p + 1e-10) for p in calibrated_probs.values())

        return {
            'max_confidence_change': max(abs(change) for change in changes.values()) if changes else 0,
            'most_increased_class': most_increased,
            'most_decreased_class': most_decreased,
            'entropy_change': cal_entropy - orig_entropy,
            'nv_bias_reduction': changes.get('nv', 0)  # Specifically track nv bias reduction
        }

    except Exception as e:
        logger.error(f"Error calculating calibration stats: {e}")
        return {}

# Convenience functions for different calibration strengths
def calibrate_conservative(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply conservative calibration (mild adjustment)."""
    CALIBRATION_CONFIG['prior_adjustment_strength'] = 0.2
    CALIBRATION_CONFIG['temperature'] = 1.1
    return calibrate_prediction_result(prediction_result, 'combined')

def calibrate_moderate(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply moderate calibration (balanced adjustment)."""
    CALIBRATION_CONFIG['prior_adjustment_strength'] = 0.3
    CALIBRATION_CONFIG['temperature'] = 1.2
    return calibrate_prediction_result(prediction_result, 'combined')

def calibrate_aggressive(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply aggressive calibration (strong adjustment for severe bias)."""
    CALIBRATION_CONFIG['prior_adjustment_strength'] = 0.5
    CALIBRATION_CONFIG['temperature'] = 1.4
    return calibrate_prediction_result(prediction_result, 'combined')


remove for experimentsation only - not used during prediction -->'''