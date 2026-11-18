# AirCast Streamlit dashboard starter

This dashboard reuses the trained Attention-LSTM model. It does **not** retrain the model.

## 1. Export files from Google Colab

After running the model training and evaluation cells (including the corrected Cell 296), add and run the export cell provided in the ChatGPT instructions.

It should create these two files:
- `aircast_bundle.joblib`
- `attention_lstm_state.pt`

Download both files from Colab and place them inside this package's `artifacts/` folder.

## 2. Local folder structure

```text
aircast_dashboard_starter/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── aircast_bundle.joblib
    └── attention_lstm_state.pt
```

## 3. Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

## Important research wording

The app forecasts one hour beyond the latest available timestamp in the exported dataset.
If the dataset is historical, call this a model-generated forecast from the latest available
historical observations, not a live sensor reading. To make it truly live, connect a verified
live data source separately.

## AQI note

The displayed category is a simplified PM2.5-based category indicator, not a full official
CPCB AQI calculation. Official AQI calculation requires pollutant sub-indices and applicable
averaging periods.
