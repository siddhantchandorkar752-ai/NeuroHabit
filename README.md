<div align="center">
  <h1>🧠 NeuroHabit OS</h1>
  <p><strong>Predictive Intelligence & Telemetry for Peak Cognitive Performance</strong></p>
</div>

<br/>

## 🌌 Overview
NeuroHabit OS is a state-of-the-art predictive intelligence engine designed to detect, analyze, and prevent cognitive burnout before it happens. By ingesting high-frequency behavioral telemetry (keystroke velocity, context switches, screen time, and systemic recovery), NeuroHabit calculates a real-time **Neural Fatigue Index** and deploys dynamic, targeted interventions.

Built with Python, FastAPI, XGBoost, and Streamlit, this system is engineered for scale, explainability, and absolute peak aesthetics.

## ✨ Core Features
- **Dynamic Telemetry Engine**: Computes Exponential Moving Averages (EMA), standard deviation volatility, and logarithmic behavioral drifts to capture rapid neural degradation.
- **Machine Learning Inference**: Powered by hyperparameter-tuned `XGBoost` regression models and `RandomForest` baselines, outputting bounded risk assessments with heuristic confidence scoring.
- **Explainability Radar**: Built-in 5-dimensional SHAP-style radar charts mapping the exact inputs causing focus fragmentation or sleep debt.
- **Object-Oriented Protocols**: Uses a dynamic `InterventionEngine` to generate macro-level behavioral overrides (e.g., "Critical Fatigue") and micro-level habit adjustments.

## 🚀 Architecture
The system is cleanly decoupled into a high-performance backend and a reactive frontend:
- **Core Engine (`src/`, `models/`)**: ML pipelines, EWMA feature engineering, and inference scripts mapping raw JSON streams into predictive vectors.
- **FastAPI Backend (`api/main.py`)**: Asynchronous REST API managing model lifespans in memory, strictly validated using Pydantic, and executing distributed background training.
- **Streamlit Frontend (`frontend/app.py`)**: A gorgeous, dark-themed command center featuring glassmorphism, completely transparent Plotly components, and massive typography.

## 💻 Installation & Usage

### 1. Initialize the Environment
Ensure you have Python 3.10+ installed.
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Ignite the Backend API
Start the FastAPI server. This API orchestrates the machine learning lifespans.
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*(You can monitor endpoints interactively at `http://localhost:8000/docs`)*

### 3. Launch the NeuroHabit Dashboard
Open a new terminal, activate your virtual environment, and run:
```bash
streamlit run frontend/app.py
```
Navigate to the provided `localhost` port to access the intelligence matrix.

## 🧪 Testing
The architecture is secured by a functional pytest suite mimicking client telemetry injections.
```bash
pytest tests/
```

---
*Developed for sustaining peak neural efficiency.*
