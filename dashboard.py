import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ── LOAD MODEL AND SCALER ─────────────────────────────────────────────


@st.cache_resource
def load_model():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('model_info.json', 'r') as f:
        info = json.load(f)
    return model, scaler, info


model, scaler, info = load_model()

WINDOW_SIZE = info['window_size']
FEATURE_NAMES = info['feature_names']

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quadruped Robot Health Monitor",
    page_icon="🤖",
    layout="wide"
)

# ── TITLE ─────────────────────────────────────────────────────────────
st.title("🤖 Quadruped Robot Anomaly Detection Dashboard")
st.markdown("**Real-time sensor monitoring and fault detection system**")
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Upload sensor data to begin analysis")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="CSV must have same sensor columns as training data"
)

st.sidebar.divider()
st.sidebar.markdown("**Model Info**")
st.sidebar.success("✅ XGBoost Model Loaded")
st.sidebar.info(f"Window Size: {WINDOW_SIZE}")
st.sidebar.info(f"Features: {len(FEATURE_NAMES)}")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────


def create_sliding_windows(X_scaled, window_size=50):
    X_windows = []
    indices = []
    for i in range(len(X_scaled) - window_size):
        window = X_scaled[i:i+window_size].flatten()
        X_windows.append(window)
        indices.append(i + window_size)
    return np.array(X_windows), indices


def calculate_health_score(predictions):
    # Health score = % of windows that are normal
    normal_pct = (predictions == 0).sum() / len(predictions) * 100
    return round(normal_pct, 1)


def get_health_color(score):
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    else:
        return "🔴"


# ── MAIN DASHBOARD ────────────────────────────────────────────────────
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file, sep=';')

    # Drop datetime if present
    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])
    if 'anomaly' in df.columns:
        true_labels = df['anomaly'].values
        df = df.drop(columns=['anomaly'])
    if 'changepoint' in df.columns:
        df = df.drop(columns=['changepoint'])

    # Keep only feature columns
    df = df[FEATURE_NAMES]
    df = df.dropna()

    # Scale
    X_scaled = scaler.transform(df)

    # Sliding windows
    X_windows, indices = create_sliding_windows(X_scaled, WINDOW_SIZE)

    # Predict
    predictions = model.predict(X_windows)
    anomaly_indices = [indices[i]
        for i in range(len(predictions)) if predictions[i] == 1]

    # Health score
    health_score = calculate_health_score(predictions)
    health_icon = get_health_color(health_score)

    # ── TOP METRICS ROW ───────────────────────────────────────────────
    st.subheader("📊 System Health Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"{health_icon} Health Score",
            value=f"{health_score}%",
            delta="Normal" if health_score >= 80 else "Degraded"
        )
    with col2:
        st.metric(
            label="Total Windows",
            value=len(predictions)
        )
    with col3:
        st.metric(
            label="🔴 Anomalies Detected",
            value=int(predictions.sum()),
            delta=f"{round(predictions.sum()/len(predictions)*100, 1)}% of data"
        )
    with col4:
        st.metric(
            label="✅ Normal Windows",
            value=int((predictions == 0).sum())
        )

    st.divider()

    # ── HEALTH SCORE GAUGE ────────────────────────────────────────────
    st.subheader("🏥 Robot Health Score")

    fig_gauge, ax = plt.subplots(figsize=(8, 2))
    color = 'green' if health_score >= 80 else 'orange' if health_score >= 60 else 'red'
    ax.barh(['Health'], [health_score], color=color, height=0.4)
    ax.barh(['Health'], [100], color='lightgrey', height=0.4, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Health Score (%)')
    ax.set_title(f'Overall Health: {health_score}%',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=80, color='green', linestyle='--',
               alpha=0.5, label='Good threshold')
    ax.axvline(x=60, color='orange', linestyle='--',
               alpha=0.5, label='Warning threshold')
    ax.legend(loc='lower right')
    st.pyplot(fig_gauge)

    st.divider()

    # ── ANOMALY TIMELINE ──────────────────────────────────────────────
    st.subheader("📈 Anomaly Detection Timeline")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(indices, predictions, color='blue', alpha=0.4, linewidth=0.8)
    ax.fill_between(indices, predictions, alpha=0.3,
                    color='red', label='Anomaly')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Prediction (0=Normal, 1=Anomaly)')
    ax.set_title('Anomaly Detection Over Time')
    ax.legend()
    st.pyplot(fig)

    st.divider()

    # ── SENSOR READINGS ───────────────────────────────────────────────
    st.subheader("📡 Sensor Readings")

    selected_sensor = st.selectbox(
        "Select sensor to visualize:", FEATURE_NAMES)

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(df[selected_sensor].values, color='steelblue',
             linewidth=0.8, label=selected_sensor)

    # Highlight anomaly regions
    for idx in anomaly_indices:
        ax2.axvspan(idx-WINDOW_SIZE, idx, alpha=0.1, color='red')

    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Sensor Value')
    ax2.set_title(f'{selected_sensor} — Red regions = detected anomalies')
    ax2.legend()
    st.pyplot(fig2)

    st.divider()

    # ── RAW PREDICTIONS TABLE ─────────────────────────────────────────
    st.subheader("📋 Prediction Summary")

    results_df = pd.DataFrame({
        'Window End Index': indices,
        'Prediction': ['🔴 ANOMALY' if p == 1 else '✅ NORMAL' for p in predictions]
    })

    # Show only anomalies
    show_anomalies_only = st.checkbox("Show anomalies only", value=False)
    if show_anomalies_only:
        st.dataframe(results_df[results_df['Prediction']
                     == '🔴 ANOMALY'], use_container_width=True)
    else:
        st.dataframe(results_df, use_container_width=True)

else:
    # ── DEFAULT SCREEN ────────────────────────────────────────────────
    st.info("👈 Upload a CSV file from the sidebar to begin analysis")

    st.markdown("""
    ### How to use this dashboard:
    1. **Upload** a CSV sensor file using the sidebar
    2. **View** the overall health score of the robot
    3. **Explore** which sensors triggered anomalies
    4. **Review** the full prediction timeline
                
    ### Expected CSV format:
datetime;Accelerometer1RMS;Accelerometer2RMS;Current;
Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS
""")
