import pandas as pd
import numpy as np
from typing import Optional

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the NeuroHabit burnout prediction model.
    Upgraded to use Exponential Moving Averages (EMA) and robust drift handling.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # 1. Robust Data Validation
    # Drop rows with corrupted/missing essential metrics
    essential_cols = ['user_id', 'date', 'screen_time_minutes', 'keystroke_speed_wpm', 'stress_level']
    df = df.dropna(subset=[c for c in essential_cols if c in df.columns])
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['user_id', 'date']).reset_index(drop=True)
    
    # Time-based features
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Advanced Rolling Features using EMA (Exponential Moving Average)
    # EMA gives more weight to recent days
    if 'sleep_hours_last_night' in df.columns:
        df['sleep_3d_avg'] = df.groupby('user_id')['sleep_hours_last_night'].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        df['sleep_3d_std'] = df.groupby('user_id')['sleep_hours_last_night'].transform(lambda x: x.rolling(3, min_periods=2).std()).fillna(0)
    
    if 'screen_time_minutes' in df.columns:
        df['screen_time_3d_avg'] = df.groupby('user_id')['screen_time_minutes'].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        
    if 'stress_level' in df.columns:
        df['stress_3d_avg'] = df.groupby('user_id')['stress_level'].transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        df['stress_7d_avg'] = df.groupby('user_id')['stress_level'].transform(lambda x: x.ewm(span=7, min_periods=1).mean())
        df['stress_3d_std'] = df.groupby('user_id')['stress_level'].transform(lambda x: x.rolling(3, min_periods=2).std()).fillna(0)

    # 3. Behavioral drift (Log-Scaled Ratios to prevent explosive variance)
    if 'keystroke_speed_wpm' in df.columns:
        df['wpm_7d_avg'] = df.groupby('user_id')['keystroke_speed_wpm'].transform(lambda x: x.ewm(span=7, min_periods=1).mean())
        # log1p transformation for safer drift ratios
        df['wpm_drift'] = np.log1p(df['keystroke_speed_wpm']) / np.log1p(df['wpm_7d_avg'] + 1e-5)
    
    if 'task_switch_frequency_per_hr' in df.columns:
        df['task_switch_7d_avg'] = df.groupby('user_id')['task_switch_frequency_per_hr'].transform(lambda x: x.ewm(span=7, min_periods=1).mean())
        df['task_switch_drift'] = np.log1p(df['task_switch_frequency_per_hr']) / np.log1p(df['task_switch_7d_avg'] + 1e-5)
    
    # Fill any remaining NaNs gracefully
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    import os
    if os.path.exists("data/dataset.csv"):
        df_raw = pd.read_csv("data/dataset.csv")
        df_features = engineer_features(df_raw)
        df_features.to_csv("data/features.csv", index=False)
        print("Feature engineering complete. Saved to data/features.csv")
    else:
        print("data/dataset.csv not found. Run generate_data.py first.")
