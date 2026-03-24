import joblib
import pandas as pd
import os
import logging
import numpy as np
from typing import Dict, Any, Tuple

# Configure standard logger
logger = logging.getLogger("BurnoutPredictor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class BurnoutPredictor:
    def __init__(self, model_dir="models/"):
        self.model_path = os.path.join(model_dir, "xgboost_model.pkl")
        self.features_path = os.path.join(model_dir, "feature_columns.pkl")
        
        self.model = None
        self.feature_cols = None
        self._load_model()
        
    def _load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.features_path):
                self.model = joblib.load(self.model_path)
                self.feature_cols = joblib.load(self.features_path)
                logger.info("Successfully loaded XGBoost model and feature schema.")
            else:
                logger.warning(f"Model artifacts not found in {self.model_path}. Run training pipeline first.")
        except Exception as e:
            logger.error(f"Failed to load model architecture: {e}")
            self.model = None
            
    def predict(self, user_features: dict) -> Tuple[float, float]:
        """
        Predict burnout score (0-100) and return a pseudo-confidence score.
        Returns: (score, confidence)
        """
        if self.model is None or self.feature_cols is None:
            logger.warning("Model not initialized. Returning baseline heuristic prediction.")
            return (50.0, 0.0) # 0 confidence
            
        try:
            # Create a single-row dataframe
            df = pd.DataFrame([user_features])
            
            # Ensure all required features are present
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0.0  # Default value for missing features
                    
            # Order columns identically to training phase
            X = df[self.feature_cols]
            
            # Predict
            score = self.model.predict(X)[0]
            score_bounded = max(0.0, min(100.0, float(score)))

            # Calculate pseudo-confidence based on missing critical data arrays
            missing_critical = sum(1 for col in ['screen_time_minutes', 'sleep_3d_avg', 'wpm_drift'] if col not in user_features)
            base_confidence = 0.95
            confidence = max(0.0, base_confidence - (missing_critical * 0.15))
            
            return (score_bounded, confidence)

        except Exception as e:
            logger.error(f"Inference Engine Critical Failure: {e}")
            return (50.0, 0.0)

if __name__ == "__main__":
    predictor = BurnoutPredictor()
    dummy_data = {
        "screen_time_minutes": 450.0,
        "keystroke_speed_wpm": 60.0,
        "sleep_hours_last_night": 5.5,
        "task_switch_frequency_per_hr": 12.0,
        "break_frequency_per_hr": 0.5,
        "day_of_week": 3,
        "is_weekend": 0,
        "sleep_3d_avg": 6.0,
        "screen_time_3d_avg": 400.0,
        "stress_3d_avg": 60.0,
        "stress_7d_avg": 55.0,
        "wpm_7d_avg": 75.0,
        "wpm_drift": 0.8,
        "task_switch_7d_avg": 5.0,
        "task_switch_drift": 2.4
    }
    score, conf = predictor.predict(dummy_data)
    print(f"Predicted Burnout Score: {score:.2f} (Confidence: {conf*100:.1f}%)")
