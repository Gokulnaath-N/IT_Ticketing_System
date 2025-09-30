import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
os.makedirs('outputs', exist_ok=True)
logging.basicConfig(filename='outputs/data_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_merge_data(data_folder: str = 'data') -> pd.DataFrame:
    try:
        files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if len(files) == 0:
            raise FileNotFoundError('No CSV files found in data folder')
        if len(files) != 5:
            logging.warning(f'Expected 5 CSV files, found {len(files)}')
        dfs = []
        for file in files:
            file_path = os.path.join(data_folder, file)
            try:
                df = pd.read_csv(file_path, parse_dates=['creation_timestamp', 'closure_timestamp'], encoding_errors='replace')
                logging.info(f'Loaded {file} with shape {df.shape}')
                dfs.append(df)
            except Exception as e:
                logging.error(f'Error loading {file}: {e}')
        if not dfs:
            raise ValueError('No dataframes loaded from CSVs')
        merged_df = pd.concat(dfs, ignore_index=True)
        if 'ticket_id' in merged_df.columns:
            before = len(merged_df)
            merged_df.drop_duplicates(subset=['ticket_id'], inplace=True)
            logging.info(f'Dropped {before - len(merged_df)} duplicate tickets by ticket_id')
        else:
            merged_df.drop_duplicates(inplace=True)
        logging.info(f'Merged shape: {merged_df.shape}')
        return merged_df
    except Exception as e:
        logging.error(f'Error in loading: {e}')
        raise


def check_dataset(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print(df.head(5))
        print(df.info())
        print(df.describe(include='all'))
        logging.info('Dataset summary checked')
        null_counts = df.isnull().sum()
        if null_counts.any():
            logging.warning(f'Missing values: {null_counts[null_counts > 0]}')
        if 'resolution_time_hours' not in df.columns and {'creation_timestamp', 'closure_timestamp'}.issubset(df.columns):
            df['resolution_time_hours'] = (df['closure_timestamp'] - df['creation_timestamp']).dt.total_seconds() / 3600
            df = df[df['resolution_time_hours'] > 0]
            logging.info('Calculated resolution time')
        return df
    except Exception as e:
        logging.error(f'Error in check_dataset: {e}')
        raise


def filter_needed_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed_cols = ['ticket_id', 'creation_timestamp', 'closure_timestamp', 'priority', 'category', 'description', 'user_department', 'resolution_time_hours']
    optional_cols = ['severity', 'channel', 'user_role', 'satisfaction_scores', 'error_codes', 'escalation_history', 'historical_volume']
    available = [col for col in needed_cols if col in df.columns]
    if len(available) < len(needed_cols):
        logging.warning('Some columns missing, using available')
    passthrough = [col for col in optional_cols if col in df.columns]
    keep_cols = list(dict.fromkeys(available + passthrough))
    return df[keep_cols]


def main():
    df = load_and_merge_data('data')
    df = check_dataset(df)
    df = filter_needed_columns(df)
    df.to_csv('data/merged_tickets.csv', index=False)
    logging.info('Data prep complete')
    print('Data preparation completed. Saved to data/merged_tickets.csv')


if __name__ == '__main__':
    main()


