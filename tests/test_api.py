from fastapi.testclient import TestClient
from api.main import app

# Create a TestClient using our FastAPI application
client = TestClient(app)

def test_predict_burnout():
    """Test the /predict endpoint with a valid payload."""
    payload = {
        "user_id": "test_user_001",
        "screen_time_minutes": 150.0,
        "keystroke_speed_wpm": 60.0,
        "sleep_hours_last_night": 7.5,
        "task_switch_frequency_per_hr": 10.0,
        "break_frequency_per_hr": 2.0,
        "day_of_week": 2,
        "is_weekend": 0,
        "sleep_3d_avg": 7.0,
        "screen_time_3d_avg": 160.0,
        "stress_3d_avg": 3.0,
        "stress_7d_avg": 3.5,
        "wpm_7d_avg": 58.0,
        "wpm_drift": 2.0,
        "task_switch_7d_avg": 12.0,
        "task_switch_drift": -2.0
    }
    
    response = client.post("/predict", json=payload)
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Parse the JSON response
    data = response.json()
    
    # Assert that we get the expected fields back
    assert data["user_id"] == "test_user_001"
    assert "burnout_score" in data
    assert "interventions" in data
    assert isinstance(data["interventions"], list)

def test_trigger_training():
    """Test the /train endpoint."""
    response = client.post("/train")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Training job triggered" in data["message"]

def test_get_user_profile():
    """Test the /user-profile endpoint."""
    # Depending on whether data exists, this could be 404 or 200.
    response = client.get("/user-profile/test_user_999")
    
    # Both 404 and 200 are acceptable depending on DB state.
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert "recent_history" in data
