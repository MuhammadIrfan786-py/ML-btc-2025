import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# --- 1. Custom Evaluation Metrics (re-defined for Streamlit app) ---
# These functions are included here to make the app self-contained.
def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_true (pd.Series): The actual values.
        y_pred (pd.Series): The predicted values.
        
    Returns:
        float: The SMAPE score.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle cases where denominator is zero
    safe_denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(numerator / safe_denominator) * 100

def mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true (pd.Series): The actual values.
        y_pred (pd.Series): The predicted values.
        
    Returns:
        float: The MAPE score.
    """
    # Avoid division by zero by checking for it
    y_true_clean = y_true.replace(0, np.nan)
    return np.nanmean(np.abs((y_true_clean - y_pred) / y_true_clean)) * 100

# --- 2. Data Loading and Model Training with Caching ---
# Caching is crucial for performance in Streamlit.
# @st.cache_data ensures the data is only loaded once.
@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the BTC data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Assuming the columns are already named 'ds' and 'y'
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# @st.cache_resource ensures the model is trained only once.
@st.cache_resource
def train_model(df):
    """Initializes, fits, and evaluates the Prophet model."""
    with st.spinner("Training model... this may take a moment."):
        # Initialize the Prophet model with the same seasonalities as your pipeline
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Fit the model to the data
        model.fit(df)

        # Perform cross-validation to get the performance metrics
        try:
            df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
            cv_mape = mape(df_cv['y'], df_cv['yhat'])
            cv_smape = smape(df_cv['y'], df_cv['yhat'])
        except ValueError:
            # Handle cases with insufficient data for cross-validation
            st.warning("Not enough data for cross-validation with a 730-day initial period. Skipping performance metrics.")
            cv_mape, cv_smape = None, None

        return model, cv_mape, cv_smape

# --- 3. Main Streamlit Application UI ---
st.set_page_config(
    page_title="BTC Prophet Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Bitcoin (BTC) Price Forecast with Prophet")
st.markdown("This app uses a **Prophet** model with monthly seasonality to forecast BTC prices. "
            "The model's performance is evaluated using cross-validation.")

# Construct the file path relative to the streamlit_app directory
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "..", "data", "processed", "btc_data_processed.csv")

# Load data and train the model
df = load_data(file_path)
if df is not None:
    model, mape_score, smape_score = train_model(df)

    # Display performance metrics
    st.header("Model Performance (30-day Horizon)")
    col1, col2 = st.columns(2)
    with col1:
        if mape_score is not None:
            st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape_score:.2f}%")
        else:
            st.info("MAPE score not available.")
    with col2:
        if smape_score is not None:
            st.metric("Symmetric MAPE (SMAPE)", f"{smape_score:.2f}%")
        else:
            st.info("SMAPE score not available.")

    # User input for forecast horizon
    st.header("Forecast")
    periods = st.slider(
        "Select number of days to forecast:",
        min_value=30,
        max_value=365,
        value=90,
        step=30,
        help="The forecast period starts from the last date in your dataset."
    )

    # Generate the forecast
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Plot the forecast
    st.subheader("Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    
    # Plot the components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

