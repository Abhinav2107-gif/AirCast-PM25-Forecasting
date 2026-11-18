import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "artifacts"
BUNDLE_PATH = ARTIFACT_DIR / "aircast_bundle.joblib"
MODEL_PATH = ARTIFACT_DIR / "attention_lstm_state.pt"

st.set_page_config(
    page_title="AirCast | PM2.5 Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

class AttentionLSTM(nn.Module):
    """Must match the architecture used in the training notebook."""
    def __init__(self, input_size=12, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)
        context = self.dropout(context)
        output = self.fc(context)
        return output.squeeze(-1), weights

@st.cache_resource
def load_assets():
    if not BUNDLE_PATH.exists() or not MODEL_PATH.exists():
        return None, None, "Required model files are missing."
    bundle = joblib.load(BUNDLE_PATH)
    device = torch.device("cpu")
    model = AttentionLSTM(input_size=len(bundle["features"]))
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return bundle, model, None

def aqi_category(pm25):
    # India CPCB PM2.5 concentration breakpoints, simplified display category.
    if pm25 <= 30:
        return "Good"
    if pm25 <= 60:
        return "Satisfactory"
    if pm25 <= 90:
        return "Moderately polluted"
    if pm25 <= 120:
        return "Poor"
    if pm25 <= 250:
        return "Very poor"
    return "Severe"

st.title("🌫️ AirCast")
st.caption("Machine-learning-based short-term PM2.5 forecasting system")
st.info(
    "Forecasts are generated from the latest records bundled with the trained "
    "research model. This is not a live sensor feed unless you connect a live data source."
)

bundle, model, error = load_assets()
if error:
    st.error(error)
    st.markdown(
        "Add `aircast_bundle.joblib` and `attention_lstm_state.pt` to the "
        "`artifacts/` folder. Follow `README.md` to export them from your Colab notebook."
    )
    st.stop()

features = bundle["features"]
city_sequences = bundle["city_sequences"]
history = bundle["history"]
metrics = bundle["metrics"]
test_frame = bundle.get("test_predictions", pd.DataFrame())

st.sidebar.header("Forecast controls")
cities = sorted(city_sequences.keys())
if not cities:
    st.error("No city has a valid 24-hour input sequence in the exported artifacts.")
    st.stop()

selected_city = st.sidebar.selectbox("Select city", cities)
seq_info = city_sequences[selected_city]
sequence = np.asarray(seq_info["sequence"], dtype=np.float32)

if sequence.shape != (24, len(features)):
    st.error(
        f"Invalid sequence shape for {selected_city}: {sequence.shape}. "
        f"Expected (24, {len(features)})."
    )
    st.stop()

with torch.no_grad():
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    pred_scaled, attention = model(x)
    pred_pm25 = float(
        bundle["target_scaler"].inverse_transform(
            pred_scaled.cpu().numpy().reshape(-1, 1)
        )[0, 0]
    )
    attention_weights = attention.cpu().numpy().flatten()

# Physical non-negativity display guard; retain raw model output for transparency.
pred_display = max(0.0, pred_pm25)
forecast_time = pd.to_datetime(seq_info["last_timestamp"]) + pd.Timedelta(hours=1)

st.subheader(f"Forecast for {selected_city}")
if pred_pm25 < 0:
    st.warning(
        f"The raw model output was {pred_pm25:.3f}. Displayed forecast is clipped "
        "to zero because negative PM2.5 concentration is not physically meaningful."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Next-hour PM2.5 forecast", f"{pred_display:.2f} µg/m³")
c2.metric("Forecast category", aqi_category(pred_display))
c3.metric("Forecast time", forecast_time.strftime("%d %b %Y, %H:%M"))
c4.metric("Input window", "Previous 24 hours")

st.caption(
    f"Last input observation: {pd.to_datetime(seq_info['last_timestamp']).strftime('%d %b %Y, %H:%M')} "
    f"| Forecast horizon: 1 hour"
)

left, right = st.columns([1.6, 1])
with left:
    city_history = history[history["City"].astype(str) == selected_city].copy()
    if not city_history.empty:
        city_history["Datetime"] = pd.to_datetime(city_history["Datetime"])
        city_history = city_history.sort_values("Datetime").tail(168)
        fig = px.line(
            city_history,
            x="Datetime",
            y="PM2.5",
            title="Recent PM2.5 history (up to last 168 available hours)",
            labels={"PM2.5": "PM2.5 (µg/m³)", "Datetime": "Date and time"},
        )
        fig.add_hline(
            y=pred_display,
            line_dash="dash",
            annotation_text=f"Next-hour forecast: {pred_display:.2f}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical PM2.5 series was exported for this city.")

with right:
    attn_df = pd.DataFrame({
        "Hour before forecast": np.arange(24, 0, -1),
        "Attention weight": attention_weights,
    })
    fig_attn = px.bar(
        attn_df,
        x="Hour before forecast",
        y="Attention weight",
        title="Attention over the 24 input time steps",
    )
    st.plotly_chart(fig_attn, use_container_width=True)
    st.caption("Attention weights indicate the model's learned time-step weighting.")

st.divider()
st.subheader("Model performance on the held-out test set")
m1, m2, m3 = st.columns(3)
m1.metric("Attention-LSTM MAE", f"{metrics['attention_lstm']['MAE']:.4f}")
m2.metric("Attention-LSTM RMSE", f"{metrics['attention_lstm']['RMSE']:.4f}")
m3.metric("Attention-LSTM R²", f"{metrics['attention_lstm']['R2']:.4f}")

comparison = pd.DataFrame(metrics["comparison"])
st.dataframe(comparison, use_container_width=True, hide_index=True)

if not test_frame.empty and {"Actual", "Predicted"}.issubset(test_frame.columns):
    sample = test_frame.head(1000).copy()
    sample["Test sample"] = np.arange(len(sample))
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=sample["Test sample"], y=sample["Actual"],
        mode="lines", name="Actual"
    ))
    fig_test.add_trace(go.Scatter(
        x=sample["Test sample"], y=sample["Predicted"],
        mode="lines", name="Predicted"
    ))
    fig_test.update_layout(
        title="Actual vs predicted PM2.5 (first 1,000 exported test samples)",
        xaxis_title="Test sample",
        yaxis_title="PM2.5 (µg/m³)",
    )
    st.plotly_chart(fig_test, use_container_width=True)

st.caption(
    "Research prototype. The dashboard presents a one-hour-ahead forecast using "
    "the trained Attention-LSTM model and exported preprocessing artifacts. "
    "It does not claim real-time monitoring without a connected live sensor/API."
)
