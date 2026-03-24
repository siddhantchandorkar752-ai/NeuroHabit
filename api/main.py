import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
import os

from models.predict import BurnoutPredictor
from src.interventions import InterventionEngine
from models.train import train_models

# Configure module logger
logger = logging.getLogger("NeuroHabitAPI")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# Global instances
predictor = None
intervention_engine = InterventionEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models
    logger.info("Initializing neural weights and prediction engine...")
    global predictor
    predictor = BurnoutPredictor(model_dir="models/")
    yield
    # Shutdown: Clean up resources if any
    logger.info("Shutting down API and releasing resources...")

app = FastAPI(
    title="NeuroHabit OS", 
    description="Predictive Cognitive Burnout Prevention Engine",
    version="2.0.0",
    lifespan=lifespan
)

# --- Pydantic Schemas ---
class FeaturesPayload(BaseModel):
    user_id: str
    screen_time_minutes: float
    keystroke_speed_wpm: float
    sleep_hours_last_night: float
    task_switch_frequency_per_hr: float
    break_frequency_per_hr: float
    day_of_week: int
    is_weekend: int
    sleep_3d_avg: float
    screen_time_3d_avg: float
    stress_3d_avg: float
    stress_7d_avg: float
    wpm_7d_avg: float
    wpm_drift: float
    task_switch_7d_avg: float
    task_switch_drift: float

class PredictionResponse(BaseModel):
    user_id: str
    burnout_score: float
    confidence_score: float
    interventions: List[Dict[str, Any]]

class UserProfileResponse(BaseModel):
    user_id: str
    recent_history: List[Dict[str, Any]]

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
async def predict_burnout(payload: FeaturesPayload):
    features_dict = payload.model_dump()
    user_id = features_dict.pop("user_id", "Unknown")
    
    try:
        # Predict using global ML instance
        score, confidence = predictor.predict(features_dict)
        
        # Generate dynamic protocol interventions
        interventions = intervention_engine.generate(features_dict, score)
        
        return PredictionResponse(
            user_id=user_id,
            burnout_score=round(score, 2),
            confidence_score=round(confidence, 2),
            interventions=interventions
        )
    except Exception as e:
        logger.error(f"Prediction fault: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference engine failure.")

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Trigger a model retraining job asynchronously.
    """
    logger.info("Received request to recompile neural weights.")
    background_tasks.add_task(train_models, data_path="data/features.csv", model_dir="models/")
    return {"message": "Training sequence queued. Monitor server logs for completion."}

@app.get("/user-profile/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(user_id: str):
    """
    Retrieve historical telemetry array for a specific entity.
    """
    data_path = "data/features.csv"
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Telemetry database unreachable.")
        
    try: # Need to catch potential parsing errors
        df = pd.read_csv(data_path)
        user_data = df[df['user_id'] == user_id]
        
        if user_data.empty:
            raise HTTPException(status_code=404, detail=f"Entity {user_id} not found in telemetry stream.")
            
        recent_data = user_data.tail(7).to_dict(orient="records")
        return UserProfileResponse(user_id=user_id, recent_history=recent_data)
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=500, detail="Database empty or corrupt.")
    except Exception as e:
        logger.error(f"Database read fault: {str(e)}")
        raise HTTPException(status_code=500, detail="Matrix query failed.")
