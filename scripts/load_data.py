"""
Load Airbnb San Diego listings dataset.

This module loads the listings.csv file from the /data directory,
selects relevant columns, and cleans the price column.
"""

import pandas as pd
from pathlib import Path
import re


def load_data(data_dir='data', filename='listings.csv'):
    """
    Load Airbnb listings CSV file and select relevant columns.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing listings.csv
    filename : str
        Name of the CSV file (default: listings.csv)
        
    Returns:
    --------
    pd.DataFrame
        Raw dataframe with selected columns
    """
    # Get the project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    data_path = project_root / data_dir / filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}\n"
            f"Please download listings.csv for San Diego (or use your own dataset)\n"
            f"and place it in {project_root / data_dir}/"
        )
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Select relevant columns for modeling
    relevant_columns = [
        'price',
        'bedrooms',
        'bathrooms',
        'accommodates',
        'room_type',
        'number_of_reviews',
        'review_scores_rating',
        'availability_365',
        'latitude',
        'longitude',
        'amenities'
    ]
    
    # Check which columns exist in the dataset
    available_columns = [col for col in relevant_columns if col in df.columns]
    missing_columns = [col for col in relevant_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following columns are missing: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()[:20]}...")  # Show first 20
    
    # Select only available columns
    df = df[available_columns].copy()
    
    # Clean price column (remove $, convert to float)
    if 'price' in df.columns:
        print("Cleaning price column...")
        df['price'] = df['price'].astype(str).str.replace('$', '', regex=False)
        df['price'] = df['price'].str.replace(',', '', regex=False)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    print(f"\nSelected dataset shape: {df.shape}")
    print(f"Selected columns: {df.columns.tolist()}")
    
    return df


if __name__ == '__main__':
    # Test the function
    df = load_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
