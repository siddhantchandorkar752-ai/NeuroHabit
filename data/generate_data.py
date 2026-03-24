import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(num_users=100, days_per_user=30, output_path="data/dataset.csv"):
    """
    Generate a synthetic dataset for NeuroHabit predicting cognitive burnout (stress_level).
    """
    print(f"Generating data for {num_users} users over {days_per_user} days...")
    
    np.random.seed(42)
    records = []
    
    start_date = datetime(2025, 1, 1)
    
    for user_id in range(1, num_users + 1):
        # User baseline characteristics
        base_stress = np.random.uniform(20, 50)
        base_sleep = np.random.normal(7.5, 1.0)
        base_wpm = np.random.normal(70, 15)
        base_screen_time = np.random.normal(300, 60) # minutes
        
        current_date = start_date
        
        # Simulate temporal dynamics (e.g., fatigue builds up over consecutive days)
        accumulated_fatigue = 0
        
        for day in range(days_per_user):
            # Weekend effect (less screen time, more sleep, less stress)
            is_weekend = current_date.weekday() >= 5
            
            # Daily variations
            sleep_hours = max(2.0, min(12.0, np.random.normal(base_sleep + (1.5 if is_weekend else 0), 1.0)))
            
            # If poor sleep, fatigue increases
            if sleep_hours < 6.5:
                accumulated_fatigue += np.random.uniform(2, 5)
            elif sleep_hours > 7.5:
                accumulated_fatigue = max(0, accumulated_fatigue - np.random.uniform(2, 5))
                
            screen_time = max(30.0, min(800.0, np.random.normal(base_screen_time - (120 if is_weekend else 0), 45)))
            
            # Behavioral metrics affected by fatigue
            # WPM drops as fatigue goes up
            keystroke_wpm = max(10.0, base_wpm - (accumulated_fatigue * 0.3) + np.random.normal(0, 5))
            
            # Task switching increases as focus drops (fatigue)
            task_switch_freq = max(1.0, np.random.normal(5 + (accumulated_fatigue * 0.2), 2))
            
            # Break frequency might decrease when stressed but increase when not
            break_freq = max(0.1, np.random.normal(2.0 - (accumulated_fatigue * 0.05), 0.5))
            
            # Calculate target variable: stress_level (0-100)
            # Stress is a function of: base_stress + fatigue + high screen time + low sleep + noise
            stress_components = (
                base_stress * 0.3 +
                accumulated_fatigue * 2.0 +
                (screen_time / 60) * 1.5 +
                (8.0 - sleep_hours) * 3.0 +
                task_switch_freq * 1.0 -
                break_freq * 2.0
            )
            
            # Add some non-linear weekly buildup (e.g., Friday is more stressful than Monday)
            if not is_weekend:
                stress_components += current_date.weekday() * 2.0
            else:
                stress_components -= 10.0 # Relief on weekends
                
            stress_level = max(0.0, min(100.0, stress_components + np.random.normal(0, 5)))
            
            records.append({
                "user_id": f"U{user_id:04d}",
                "date": current_date.strftime("%Y-%m-%d"),
                "screen_time_minutes": round(screen_time, 2),
                "keystroke_speed_wpm": round(keystroke_wpm, 2),
                "sleep_hours_last_night": round(sleep_hours, 2),
                "task_switch_frequency_per_hr": round(task_switch_freq, 2),
                "break_frequency_per_hr": round(break_freq, 2),
                "stress_level": round(stress_level, 2)
            })
            
            current_date += timedelta(days=1)
            
    df = pd.DataFrame(records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")
    print(df.head())
    print("\nDataset Summary Insights:")
    print(df['stress_level'].describe())

if __name__ == "__main__":
    # Generate larger dataset points for robust training
    generate_synthetic_data(num_users=200, days_per_user=60, output_path="data/dataset.csv")
