# Import necessary libraries for the Prophet model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# --- Define Custom Evaluation Metrics (MAPE and SMAPE) ---
# Prophet's performance_metrics function calculates RMSE and others.
# We'll use these custom functions for MAPE and SMAPE.

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

def main(file_path, date_column, target_column, processed_columns):
    """
    Main function to run the Prophet forecasting pipeline.
    
    Args:
        file_path (str): Path to the dataset CSV file.
        date_column (str): Name of the date column in the dataset.
        target_column (str): Name of the target value column.
        processed_columns (list): List of columns to use for training.
    """
    try:
        # --- 1. Load Pre-processed Time Series Data ---
        print("Starting training pipeline...")
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path, usecols=[date_column, target_column])
        
        # Rename columns to Prophet's required format ('ds' and 'y')
        df = df.rename(columns={date_column: 'ds', target_column: 'y'})
        
        # Convert the 'ds' column to a datetime object, which is crucial for Prophet.
        df['ds'] = pd.to_datetime(df['ds'])
        
        print("\nData loaded successfully.")
        print("DataFrame head:")
        print(df.head())
        print("-" * 30)

        # --- 2. Prophet Model Setup and Fitting ---
        # Initialize the model and tell it to look for monthly seasonality,
        # as you identified from your STL analysis.
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'  # Additive mode is often good for prices
        )
        # Add a custom monthly seasonality based on your findings. A month is ~30.5 days.
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit the model to the entire dataset
        print("Fitting the Prophet model...")
        model.fit(df)
        print("Model fitting complete.")
        print("-" * 30)

        # --- 3. Cross-Validation and Performance Evaluation ---
        # Use cross-validation to get a more reliable measure of performance.
        # We'll forecast 30 days ahead, training on a rolling window.
        print("Starting cross-validation...")
        try:
            # `initial`: The size of the initial training period
            # `period`: The interval between cutoff dates
            # `horizon`: The forecast horizon
            df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
            print("Cross-validation complete.")
            print("-" * 30)
        
            # Calculate performance metrics from the cross-validation results.
            cv_mape = mape(df_cv['y'], df_cv['yhat'])
            cv_smape = smape(df_cv['y'], df_cv['yhat'])

            print(f"Cross-Validation Results (30-day horizon):")
            print(f"  Mean Absolute Percentage Error (MAPE): {cv_mape:.2f}%")
            print(f"  Symmetric Mean Absolute Percentage Error (SMAPE): {cv_smape:.2f}%")
            print("-" * 30)
            
        except ValueError as e:
            print(f"Cross-validation failed. This is likely due to insufficient data for the specified `initial` period.")
            print(f"Error details: {e}")

        # --- 4. Making a Future Forecast ---
        # Create a future dataframe to forecast for the next 365 days
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # --- 5. Visualization ---
        # Plot the forecast
        fig1 = model.plot(forecast)
        fig1.suptitle('BTC-USD Forecast', y=1.02)
        plt.show()

        # Plot the forecast components to visualize the monthly seasonality
        fig2 = model.plot_components(forecast)
        fig2.suptitle('Forecast Components', y=1.02)
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the file is in the correct directory.")
    except KeyError as e:
        print(f"Error: A required column was not found in the dataset. Please check your `date_column` and `target_column` names.")
        print(f"KeyError details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the pipeline execution: {e}")


