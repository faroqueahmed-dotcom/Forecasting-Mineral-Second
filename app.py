import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# -----------------------------
# Data Generation
# -----------------------------
dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='MS')
countries = ['Canada', 'Australia', 'Brazil']
minerals = ['Iron', 'Copper', 'Lithium']

np.random.seed(42)
data = []
for country in countries:
    for mineral in minerals:
        trend = np.linspace(50, 200, len(dates))
        seasonal = 10 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
        noise_price = np.random.normal(0, 5, len(dates))
        noise_demand = np.random.normal(0, 3, len(dates))
        price = trend + seasonal + noise_price
        demand = trend/2 + seasonal + noise_demand
        reserves = np.linspace(1000, 500, len(dates))
        for i in range(len(dates)):
            data.append([country, mineral, dates[i], price[i], reserves[i], demand[i]])

columns = ['Country', 'Mineral', 'Date', 'Price', 'Reserves', 'Demand']
df = pd.DataFrame(data, columns=columns)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Mineral Market Dashboard")

# Filters
country = st.sidebar.selectbox("Select Country", df['Country'].unique())
mineral = st.sidebar.selectbox("Select Mineral", df['Mineral'].unique())
variable = st.sidebar.selectbox("Select Variable to Forecast", ['Price', 'Demand'])
forecast_horizon = st.sidebar.slider("Forecast horizon (months)", 12, 120, 24)
models_selected = st.sidebar.multiselect(
    "Select Forecasting Models",
    ['SARIMA', 'Auto ARIMA', 'XGBoost', 'CatBoost', 'LSTM'],
    default=['SARIMA']
)

plot_type = st.sidebar.selectbox(
    "Select Plot Type for Historical Data",
    ['Line', 'Bar', 'Scatter', 'Compare Variables']
)

filtered_df = df[(df['Country']==country) & (df['Mineral']==mineral)]

# -----------------------------
# Descriptive Statistics
# -----------------------------
st.subheader(f"Descriptive Statistics of {mineral} in {country}")
desc_cols = ['Price', 'Reserves', 'Demand']
desc_cols = [c for c in desc_cols if c in filtered_df.columns]  # Remove duplicates
st.dataframe(filtered_df[desc_cols].describe())

# -----------------------------
# Historical Data Plot
# -----------------------------
st.subheader(f"Historical Data for {mineral} in {country}")

if plot_type == 'Line':
    fig_hist = px.line(filtered_df, x='Date', y=variable, title=f"{variable} Over Time")

elif plot_type == 'Bar':
    fig_hist = px.bar(filtered_df, x='Date', y=variable, title=f"{variable} Over Time")

elif plot_type == 'Scatter':
    x_var = st.sidebar.selectbox("Select X-axis for Scatter", desc_cols)
    y_var = st.sidebar.selectbox("Select Y-axis for Scatter", desc_cols)
    fig_hist = px.scatter(filtered_df, x=x_var, y=y_var, title=f"{y_var} vs {x_var}", trendline="lowess")

elif plot_type == 'Compare Variables':
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Price'], mode='lines', name='Price'))
    fig_hist.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Demand'], mode='lines', name='Demand'))
    fig_hist.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Reserves'], mode='lines', name='Reserves'))
    fig_hist.update_layout(
        title=f"Comparison of Price, Demand, and Reserves for {mineral} in {country}",
        xaxis_title="Date",
        yaxis_title="Value"
    )

st.plotly_chart(fig_hist)



# -----------------------------
# Forecasting Functions
# -----------------------------
variable_series = filtered_df.set_index('Date')[variable]

def forecast_sarima(series, periods):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit(disp=False)
    forecast_obj = result.get_forecast(steps=periods)
    forecast_df = forecast_obj.summary_frame(alpha=0.05).reset_index()
    forecast_df.rename(columns={'index':'Date', 'mean':'mean','mean_ci_lower':'mean_ci_lower','mean_ci_upper':'mean_ci_upper'}, inplace=True)
    return forecast_df

def forecast_arima(series, periods):
    arima_model = auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
    forecast_obj, conf_int = arima_model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)
    last_date = series.index[-1]
    forecast_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=periods, freq='MS')
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_obj,
        'Lower CI': conf_int[:,0],
        'Upper CI': conf_int[:,1]
    })
    return forecast_df

def forecast_ml(series, model_type, periods):
    df_ml = series.reset_index()
    df_ml['time_index'] = np.arange(len(df_ml))
    X = df_ml[['time_index']].values
    y = df_ml[series.name].values

    if model_type == 'XGBoost':
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=500)
        model.fit(X, y)
    elif model_type == 'CatBoost':
        model = CatBoostRegressor(verbose=0, iterations=500)
        model.fit(X, y)
    else:  # LSTM
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1,1))
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1,1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        X_scaled = X.reshape((X.shape[0],1,1))
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=16, verbose=0)

    # Forecast future
    X_next = np.arange(len(series), len(series)+periods).reshape(-1,1)
    if model_type == 'LSTM':
        X_next_scaled = X_next.reshape((X_next.shape[0],1,1))
        forecast = model.predict(X_next_scaled)
        forecast = scaler.inverse_transform(forecast).flatten()
    else:
        forecast = model.predict(X_next)

    forecast_dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq='MS')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    return forecast_df

# -----------------------------
# Forecasting Plots
# -----------------------------
for model in models_selected:
    st.subheader(f"Forecasted {variable} ({model}) for {mineral} in {country}")
    
    if model == 'SARIMA':
        forecast_data = forecast_sarima(variable_series, forecast_horizon)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=variable_series.index, y=variable_series.values, mode='lines', name='Historical'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['mean'], mode='lines', name='Forecast'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['mean_ci_lower'], fill=None, mode='lines', line=dict(dash='dash'), name='Lower CI'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['mean_ci_upper'], fill='tonexty', mode='lines', line=dict(dash='dash'), name='Upper CI'))

    elif model == 'Auto ARIMA':
        forecast_data = forecast_arima(variable_series, forecast_horizon)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=variable_series.index, y=variable_series.values, mode='lines', name='Historical'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Forecast'], mode='lines', name='Forecast'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Lower CI'], fill=None, mode='lines', line=dict(dash='dash'), name='Lower CI'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Upper CI'], fill='tonexty', mode='lines', line=dict(dash='dash'), name='Upper CI'))

    else:  # ML models
        forecast_data = forecast_ml(variable_series, model, forecast_horizon)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=variable_series.index, y=variable_series.values, mode='lines', name='Historical'))
        fig_forecast.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Forecast'], mode='lines', name='Forecast'))

    st.plotly_chart(fig_forecast)

# -------------------
# Raw Data
# -------------------
with st.expander("Show Raw Data"):
    st.dataframe(filtered_df)