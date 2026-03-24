import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NeuroHabit OS", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PROFESSIONAL AESTHETICS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sleek gradient background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #16213e 0%, #0f172a 60%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Clean up default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] * {
        font-size: 1.15rem;
    }
    
    /* Headers */
    h1 {
        font-size: 3.5rem !important;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
    }
    h2, h3 {
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: 0.5px;
    }
    h3 { font-size: 1.8rem !important; }
    
    /* Subtitle */
    .subtitle {
        color: #94a3b8;
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 3.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 16px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.3s ease, border-color 0.3s ease;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    /* Shine effect for card */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.03) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        transition: 0.7s;
    }
    .metric-card:hover::before {
        left: 200%;
    }
    
    .metric-value {
        font-size: 6rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 15px 0;
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
    }
    
    .metric-label {
        color: #cbd5e1;
        font-size: 1.25rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .metric-status {
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: 3px;
        margin-top: 15px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4) !important;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6) !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        background: rgba(30, 41, 59, 0.5) !important;
        padding: 25px !important;
        margin-bottom: 20px !important;
    }
    .stAlert p {
        font-size: 1.3rem !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# Header
st.markdown("<h1>NeuroHabit OS</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predictive Intelligence for Peak Cognitive Performance</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h3 style='margin-bottom: 25px;'>Telemetry Injector</h3>", unsafe_allow_html=True)
    user_id = st.text_input("Entity ID", "test_user_001", help="Unique identifier for the subject.")
    
    st.markdown("<br/>", unsafe_allow_html=True)
    screen_time = st.slider("Screen Time (mins)", 0.0, 800.0, 300.0, help="Continuous focal time without ocular rest.")
    wpm = st.slider("Keystroke Velocity (WPM)", 10.0, 150.0, 70.0, help="Motor function indicator.")
    sleep = st.slider("Recovery Sleep (hours)", 0.0, 12.0, 7.5, help="Neural repair cycle duration.")
    task_switch = st.slider("Context Switches (per hr)", 0.0, 30.0, 5.0, help="Measure of focus fragmentation.")
    breaks = st.slider("Micro-breaks (per hr)", 0.0, 5.0, 1.0, help="Autonomic nervous system resets.")
    
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    if st.button("Compile Neural Weights"):
        with st.spinner("Executing distributed training..."):
            try:
                res = requests.post(f"{API_URL}/train")
                if res.status_code == 200:
                    st.success("Training sequence successful.")
                else:
                    st.error("Sequence failed.")
            except:
                st.error("API Connection Offline.")

# Data Prep
payload = {
    "user_id": user_id,
    "screen_time_minutes": screen_time,
    "keystroke_speed_wpm": wpm,
    "sleep_hours_last_night": sleep,
    "task_switch_frequency_per_hr": task_switch,
    "break_frequency_per_hr": breaks,
    "day_of_week": 2, "is_weekend": 0, "sleep_3d_avg": 7.0,
    "screen_time_3d_avg": 350.0, "stress_3d_avg": 40.0, "stress_7d_avg": 45.0,
    "wpm_7d_avg": 75.0, "wpm_drift": wpm / 75.0 if wpm else 1.0,
    "task_switch_7d_avg": 4.5, "task_switch_drift": task_switch / 4.5 if task_switch else 1.0
}

# Fetch Historical
df_hist = pd.DataFrame()
try:
    profile_response = requests.get(f"{API_URL}/user-profile/{user_id}")
    if profile_response.status_code == 200:
        history = profile_response.json().get("recent_history", [])
        if history:
            df_hist = pd.DataFrame(history)
except Exception:
    pass

col1, col2 = st.columns([1.1, 1.9], gap="large")

with col1:
    st.markdown("<h3 style='margin-bottom: 25px;'>Neural Inference Matrix</h3>", unsafe_allow_html=True)
    try:
        predict_response = requests.post(f"{API_URL}/predict", json=payload)
        if predict_response.status_code == 200:
            result = predict_response.json()
            score = result.get("burnout_score", 0)
            confidence = result.get("confidence_score", 0)
            interventions = result.get("interventions", [])
            
            # FIXED: Score bounds are 0-100!
            if score < 40:
                color, glow, state = "#10b981", "rgba(16, 185, 129, 0.4)", "OPTIMAL" # Emerald
            elif score < 70:
                color, glow, state = "#f59e0b", "rgba(245, 158, 11, 0.4)", "ELEVATED" # Amber
            else:
                color, glow, state = "#ef4444", "rgba(239, 68, 68, 0.4)", "CRITICAL" # Rose

            st.markdown(f"""
            <div class="metric-card" style="border-left: 8px solid {color};">
                <div class="metric-label">Predicted Burnout Risk</div>
                <div class="metric-value" style="color: {color}; text-shadow: 0 0 40px {glow};">{score:.1f}<span style="font-size:2.5rem; color: #64748b;"> / 100</span></div>
                <div class="metric-status" style="color: {color};">{state} STATE</div>
                <div style="margin-top: 25px; font-size: 1.15rem; color: #94a3b8; display: flex; justify-content: space-between;">
                    <span>Model Confidence</span>
                    <span style="color: #cbd5e1; font-weight: 600;">{confidence * 100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3 style='margin-top: 40px; margin-bottom: 25px;'>Active Action Protocols</h3>", unsafe_allow_html=True)
            for inv in interventions:
                if isinstance(inv, dict):
                    inv_type = inv.get("type", "INFO")
                    msg = inv.get("message", "Unknown protocol.")
                else:
                    msg, inv_type = str(inv), "INFO"
                    
                if inv_type == "CRITICAL":
                    st.error(f"**OVERRIDE:** {msg}", icon="🚨")
                elif inv_type == "WARNING" or inv_type == "LIFESTYLE":
                    st.warning(f"**ADVISORY:** {msg}", icon="⚠️")
                elif inv_type == "POSITIVE":
                    st.success(f"**STABLE:** {msg}", icon="✅")
                else:
                    st.info(f"**SUGGESTION:** {msg}", icon="💡")
        else:
            st.error("API Backend Offline or Error")
    except Exception as e:
        st.error(f"SYSTEM OFFLINE: Execute `uvicorn api.main:app` to establish connection. {e}")

with col2:
    st.markdown("<h3 style='margin-bottom: 25px;'>Longitudinal Telemetry & Explainability</h3>", unsafe_allow_html=True)
    
    # 1. Provide an "Explainability" Radar Chart for Professional Data Science Look
    # We map current inputs to a 0-1 risk scale
    risk_vectors = pd.DataFrame(dict(
        r=[
            min(1.0, task_switch / 15.0), 
            min(1.0, screen_time / 600.0), 
            max(0.0, 1.0 - (sleep / 8.0)), 
            max(0.0, 1.0 - (wpm / 100.0)),
            max(0.0, 1.0 - (breaks / 3.0))
        ],
        theta=['Focus Fragmentation', 'Ocular Strain', 'Sleep Debt', 'Motor Degradation', 'Recovery Deficit']
    ))
    
    fig_radar = px.line_polar(risk_vectors, r='r', theta='theta', line_close=True, template="plotly_dark")
    fig_radar.update_traces(fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', line_color='#38bdf8', line_width=3)
    fig_radar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=15, color="#cbd5e1", family="Outfit"))
        ),
        margin=dict(l=50, r=50, t=30, b=30),
        height=400
    )
    
    # Render radar chart in a card
    st.markdown("""
        <div class="metric-card" style="padding: 25px;">
            <div class="metric-label" style="margin-bottom: -10px;">Real-time Risk Vector Explainability</div>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True)


    if not df_hist.empty and 'date' in df_hist.columns:
        st.markdown("<h3 style='margin-top: 40px; margin-bottom: 25px;'>7-Day Pattern Analysis</h3>", unsafe_allow_html=True)
        fig = go.Figure()
        
        if 'stress_level' in df_hist.columns:
            fig.add_trace(go.Scatter(
                x=df_hist['date'], y=df_hist['stress_level'],
                mode='lines+markers', name='Stress Level',
                line=dict(color='#f43f5e', width=4, shape='spline'),
                marker=dict(size=12, color='#f43f5e', line=dict(width=2, color='#0f172a'))
            ))
        
        if 'wpm_drift' in df_hist.columns:
            fig.add_trace(go.Scatter(
                x=df_hist['date'], y=df_hist['wpm_drift'] * 100, 
                mode='lines+markers', name='Motor Function (%)',
                line=dict(color='#38bdf8', width=4, dash='dot', shape='spline'),
                marker=dict(size=12, color='#38bdf8', line=dict(width=2, color='#0f172a'))
            ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Outfit", color="#f8fafc", size=15),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, font=dict(size=14)),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, zeroline=False, color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#94a3b8"),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 40px;">
            <div class="metric-label">NO LONGITUDINAL DATA</div>
            <p style="color: #94a3b8; margin-top: 15px; font-size: 1.3rem;">Entity '{user_id}' has no recorded continuous telemetry.</p>
        </div>
        """, unsafe_allow_html=True)
