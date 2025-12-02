"""
Preprocess Airbnb listings data: handle missing values, encode categoricals, process amenities.

This module handles data cleaning, missing value imputation, categorical encoding,
and feature engineering for the Airbnb dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import re


def count_amenities(amenities_str):
    """
    Count the number of amenities from the amenities string.
    
    The amenities column is typically a string representation of a list.
    Example: '["TV", "WiFi", "Kitchen"]' -> 3
    
    Parameters:
    -----------
    amenities_str : str or NaN
        String representation of amenities list
        
    Returns:
    --------
    int
        Number of amenities, or 0 if invalid
    """
    if pd.isna(amenities_str):
        return 0
    
    try:
        # Try to parse as Python literal (list)
        if isinstance(amenities_str, str):
            # Remove leading/trailing quotes if present
            amenities_str = amenities_str.strip()
            if amenities_str.startswith('"') and amenities_str.endswith('"'):
                amenities_str = amenities_str[1:-1]
            
            # Try to evaluate as Python literal
            try:
                amenities_list = ast.literal_eval(amenities_str)
                if isinstance(amenities_list, list):
                    return len(amenities_list)
            except:
                pass
            
            # Fallback: count commas + 1 (rough estimate)
            if ',' in amenities_str or '[' in amenities_str:
                # Count items by splitting on common delimiters
                items = re.split(r'[,\[\]]+', amenities_str)
                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                return len(items) if items else 0
        
        return 0
    except:
        return 0


def preprocess(df):
    """
    Preprocess the Airbnb listings dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe from load_data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed dataframe ready for EDA + modeling
    """
    df = df.copy()
    
    print("Starting preprocessing...")
    print("=" * 50)
    
    # Handle missing values in numerical columns
    print("\nHandling missing values...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'price' in numerical_cols:
        numerical_cols.remove('price')  # Don't fill price with mean
    
    print(f"Numerical columns to impute: {numerical_cols}")
    
    for col in numerical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
            # Fill with median (more robust to outliers)
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    Filled with median: {median_val:.2f}")
    
    # Process amenities: convert to count
    if 'amenities' in df.columns:
        print("\nProcessing amenities column...")
        df['amenities_count'] = df['amenities'].apply(count_amenities)
        df.drop('amenities', axis=1, inplace=True)
        print(f"Amenities count range: {df['amenities_count'].min()} - {df['amenities_count'].max()}")
    
    # Encode categorical features (room_type)
    if 'room_type' in df.columns:
        print("\nEncoding categorical features...")
        print(f"Room types: {df['room_type'].value_counts().to_dict()}")
        # One-hot encode room_type (keep all categories, drop_first=False for interpretability)
        room_type_dummies = pd.get_dummies(df['room_type'], prefix='room_type', drop_first=True)
        df = pd.concat([df, room_type_dummies], axis=1)
        df.drop('room_type', axis=1, inplace=True)
        print(f"Created {len(room_type_dummies.columns)} room_type dummy variables")
    
    # Remove rows with missing price (target variable)
    initial_rows = len(df)
    df = df.dropna(subset=['price'])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"\nRemoved {removed_rows} rows with missing price ({removed_rows/initial_rows*100:.1f}%)")
    
    # Remove rows with extreme outliers in price (optional - keep for now)
    # Could add: df = df[(df['price'] > 0) & (df['price'] < df['price'].quantile(0.99))]
    
    # Ensure all remaining numerical columns are numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any remaining NaN in features with 0 (shouldn't be many after above steps)
    df = df.fillna(0)
    
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print("=" * 50)
    print(f"Final dataset shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")
    
    return df


if __name__ == '__main__':
    # Test the preprocessing
    from load_data import load_data
    
    df_raw = load_data()
    df_clean = preprocess(df_raw)
    
    print("\nPreprocessed dataframe:")
    print(df_clean.head())
    print("\nData types:")
    print(df_clean.dtypes)
