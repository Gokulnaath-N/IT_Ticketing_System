"""
Data Preparation Script for IT Support Ticket Analysis
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/logs/data_preparation.log'),
        logging.StreamHandler()
    ]
)

def load_and_merge_data(data_folder: str) -> pd.DataFrame:
    """Load and merge all CSV files in the data folder."""
    try:
        data_dir = Path(data_folder)
        data_files = list(data_dir.glob("*.csv"))
        
        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        logging.info(f"Found {len(data_files)} CSV files in {data_dir}")
        
        # Read and concatenate all CSV files
        dfs = []
        for file in data_files:
            try:
                df = pd.read_csv(file, encoding_errors='replace')
                dfs.append(df)
                logging.info(f"Loaded {file.name} with shape {df.shape}")
            except Exception as e:
                logging.error(f"Error loading {file.name}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No valid CSV files could be loaded")
            
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Clean up duplicate rows if any
        initial_rows = len(merged_df)
        merged_df.drop_duplicates(inplace=True)
        if len(merged_df) < initial_rows:
            logging.info(f"Removed {initial_rows - len(merged_df)} duplicate rows")
            
        logging.info(f"Merged data shape: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        logging.error(f"Error in load_and_merge_data: {str(e)}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the data."""
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logging.info("Handling missing values:")
            logging.info(missing[missing > 0])
            
            # Fill missing values based on column type
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna('Unknown', inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
        
        # Convert date columns if they exist
        date_columns = ['created_at', 'updated_at', 'resolved_at', 'creation_timestamp', 'closure_timestamp']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logging.info(f"Converted {col} to datetime")
                except Exception as e:
                    logging.warning(f"Could not convert {col} to datetime: {str(e)}")
        
        # Clean text columns
        text_columns = ['subject', 'description', 'body', 'resolution_notes']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Remove extra whitespace
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error in clean_data: {str(e)}")
        raise

def save_processed_data(df: pd.DataFrame, output_dir: str = None) -> None:
    """Save the processed data to a CSV file."""
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "processed")
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "processed_tickets.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {str(e)}")
        raise

def main():
    try:
        # Define base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create necessary directories
        output_dir = os.path.join(base_dir, "outputs", "logs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Load and merge data
        logging.info("Starting data preparation...")
        data_dir = os.path.join(base_dir, "data")
        df = load_and_merge_data(data_dir)
        
        # Step 2: Clean and preprocess data
        df_cleaned = clean_data(df)
        
        # Step 3: Save processed data
        save_processed_data(df_cleaned)
        
        # Basic statistics
        logging.info("\nData Preparation Summary:")
        logging.info(f"Total records: {len(df_cleaned)}")
        logging.info("\nFirst few records:")
        logging.info(df_cleaned.head().to_string())
        
        logging.info("\nData preparation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
