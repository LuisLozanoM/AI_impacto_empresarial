import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from lgb_forecast import run_forecasting_pipeline

st.set_page_config(page_title="Electric Production Forecast", layout="wide")

st.title("⚡ Electric Production Forecasting")
st.markdown("""
This app forecasts quarterly electric production using **LightGBM** and **Conformal Prediction**.
It uses the last 8 quarters (y-8 to y-1) to predict the next 4 quarters (y+1 to y+4).
""")

# Run pipeline
with st.spinner("Training models and generating forecasts..."):
    try:
        # Unpack 8 values now
        df_history, results_df, test_mape, test_mae, train_mape, train_mae, full_data_df, importance_df = run_forecasting_pipeline("Electric_Production.csv")
        st.success("Forecasting complete!")
    except Exception as e:
        st.error(f"Error running pipeline: {e}")
        st.stop()

# Metrics
st.subheader("Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test MAPE", f"{test_mape:.2%}")
col2.metric("Test MAE", f"{test_mae:.4f}")
col3.metric("Train MAPE", f"{train_mape:.2%}")
col4.metric("Train MAE", f"{train_mae:.4f}")

# Visualization
st.subheader("Forecast Visualization (Last 5 Years)")

# Filter df_history to last 5 years
# Last date in df_history
last_date = df_history.index[-1]
start_date = last_date - pd.DateOffset(years=5)
df_history_plot = df_history[df_history.index >= start_date]

# Create a figure
fig = go.Figure()

# 1. Historical Data
fig.add_trace(go.Scatter(
    x=df_history_plot.index,
    y=df_history_plot['Value'],
    mode='lines+markers',
    name='Historical Data',
    line=dict(color='blue')
))

# 2. Predictions and Intervals
# We only show predictions that fall within the plot range or future.
# H1 predictions
h1_dates = []
h1_preds = []
h1_lower = []
h1_upper = []

for idx, row in results_df.iterrows():
    anchor = row['Date_Anchor']
    h1_date = anchor + pd.DateOffset(months=3) 
    
    # Only add if it's relevant to the plot (or future)
    if h1_date >= start_date:
        h1_dates.append(h1_date)
        h1_preds.append(row['Pred_h1'])
        h1_lower.append(row['Lower_h1'])
        h1_upper.append(row['Upper_h1'])

# Add H1 Predictions
fig.add_trace(go.Scatter(
    x=h1_dates,
    y=h1_preds,
    mode='lines',
    name='1-Step Ahead Prediction',
    line=dict(color='red', dash='dash')
))

# Add Confidence Interval (Shaded)
fig.add_trace(go.Scatter(
    x=h1_dates + h1_dates[::-1],
    y=h1_upper + h1_lower[::-1],
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='90% Confidence Interval (H1)'
))

# Latest Forecast (Future)
last_row = results_df.iloc[-1]
last_anchor = last_row['Date_Anchor']
future_dates = [last_anchor + pd.DateOffset(months=3*i) for i in range(1, 5)]
future_preds = [last_row[f'Pred_h{i}'] for i in range(1, 5)]
future_lower = [last_row[f'Lower_h{i}'] for i in range(1, 5)]
future_upper = [last_row[f'Upper_h{i}'] for i in range(1, 5)]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_preds,
    mode='lines+markers',
    name='Latest 4-Step Forecast',
    line=dict(color='green', width=3)
))

# Error Bars for latest forecast
fig.add_trace(go.Scatter(
    x=future_dates + future_dates[::-1],
    y=future_upper + future_lower[::-1],
    fill='toself',
    fillcolor='rgba(0, 255, 0, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='90% CI (Latest Forecast)'
))

st.plotly_chart(fig, use_container_width=True)

st.subheader("Detailed Results")
st.dataframe(results_df)

st.subheader("Feature Importance")
st.markdown("Average feature importance across all 4 forecasting models.")
fig_imp = go.Figure(go.Bar(
    x=importance_df['Importance'],
    y=importance_df['Feature'],
    orientation='h'
))
fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_imp, use_container_width=True)

st.subheader("Training Data (Features & Targets)")
st.markdown("This table shows the features (lags, rolling stats, diffs) and targets used for training.")
st.dataframe(full_data_df)
