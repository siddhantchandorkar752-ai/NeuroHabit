import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logger = logging.getLogger("TrainingPipeline")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def train_models(data_path="data/features.csv", model_dir="models/"):
    logger.info("Initializing neural weights compilation (Training Sequence)...")
    
    if not os.path.exists(data_path):
        logger.error(f"Data pathway disrupted: {data_path} not found.")
        return False
        
    df = pd.read_csv(data_path)
    
    # Drop non-predictive columns and target
    features_to_drop = ['user_id', 'date', 'stress_level']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    
    if 'stress_level' not in df.columns:
        logger.error("Target variable 'stress_level' missing from dataset.")
        return False
        
    y = df['stress_level']
    logger.info(f"Dataset matrix locked: {X.shape[0]} samples, {X.shape[1]} telemetry streams")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Engaging XGBoost architecture...")
    
    # Fast RandomizedSearchCV for Hyperparameter tuning
    param_distributions = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0]
    }
    
    base_xgb = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    # Using small cv and iter for fast execution
    search = RandomizedSearchCV(
        base_xgb, 
        param_distributions, 
        n_iter=5, 
        scoring='neg_mean_absolute_error',
        cv=3, 
        random_state=42, 
        n_jobs=-1
    )
    
    logger.info("Executing grid search for optimal hyperparameters...")
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    
    xgb_preds = best_xgb.predict(X_test)
    
    logger.info("--- Model Viability Report ---")
    logger.info(f"Optimum Params: {search.best_params_}")
    logger.info(f"MAE: {mean_absolute_error(y_test, xgb_preds):.4f}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, xgb_preds)):.4f}")
    logger.info(f"R2 : {r2_score(y_test, xgb_preds):.4f}")
    
    # Save the best model
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "xgboost_model.pkl")
    joblib.dump(best_xgb, best_model_path)
    logger.info(f"Primary model matrix saved: {best_model_path}")
    
    # Save feature columns
    feature_cols_path = os.path.join(model_dir, "feature_columns.pkl")
    joblib.dump(list(X.columns), feature_cols_path)
    logger.info("Feature schema persisted.")
    return True

if __name__ == "__main__":
    train_models()
