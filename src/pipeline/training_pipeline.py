"""
Script to run the training pipeline for the Prophet BTC forecasting model.
:param file_path: Path to the dataset CSV file.
:param target_column: Name of the target column.
"""
from src.models.fbp import main
import os

# Columns to be used from the dataset (must include date and target)
processed_columns = ['ds', 'y']

if __name__ == "__main__":
    # Path to processed BTC data
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "..", "..", "data", "processed", "btc_data_processed.csv")
    
    # Run the main training pipeline
    main(
        file_path=file_path,
        date_column="ds",  # Name of your date column
        target_column="y",  # Name of your target column
        processed_columns=processed_columns
    )