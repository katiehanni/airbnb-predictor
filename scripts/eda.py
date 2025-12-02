"""
Exploratory Data Analysis: Generate visualizations and summary statistics for Airbnb data.

This module creates various plots and saves them to the output/figures directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def ensure_output_dir():
    """Ensure output/figures directory exists."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'output' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_price_distribution(df, output_dir):
    """
    Plot price distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    output_dir : Path
        Directory to save figures
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram
    ax1 = axes[0, 0]
    df['price'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_title('Price Distribution (Histogram)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['price'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${df["price"].mean():.2f}')
    ax1.axvline(df['price'].median(), color='green', linestyle='--', 
                label=f'Median: ${df["price"].median():.2f}')
    ax1.legend()
    
    # Plot 2: Log scale histogram
    ax2 = axes[0, 1]
    df['price'].hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_yscale('log')
    ax2.set_title('Price Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Frequency (log scale)')
    
    # Plot 3: Box plot
    ax3 = axes[1, 0]
    df.boxplot(column='price', ax=ax3, grid=False)
    ax3.set_title('Price Distribution (Box Plot)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Price ($)')
    
    # Plot 4: Q-Q plot (quantile-quantile)
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(df['price'], dist="norm", plot=ax4)
    ax4.set_title('Price Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: price_distribution.png")


def plot_price_vs_features(df, output_dir):
    """
    Plot price vs bedrooms and bathrooms.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    output_dir : Path
        Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Price vs Bedrooms
    if 'bedrooms' in df.columns:
        ax1 = axes[0]
        bedroom_groups = df.groupby('bedrooms')['price'].agg(['mean', 'median', 'count'])
        bedroom_groups = bedroom_groups[bedroom_groups['count'] >= 10]  # Only show groups with enough data
        
        x_pos = bedroom_groups.index
        ax1.bar(x_pos, bedroom_groups['mean'], alpha=0.7, color='steelblue', label='Mean')
        ax1.plot(x_pos, bedroom_groups['median'], 'ro-', label='Median', linewidth=2, markersize=8)
        ax1.set_title('Price vs Bedrooms', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Bedrooms')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price vs Bathrooms
    if 'bathrooms' in df.columns:
        ax2 = axes[1]
        bathroom_groups = df.groupby('bathrooms')['price'].agg(['mean', 'median', 'count'])
        bathroom_groups = bathroom_groups[bathroom_groups['count'] >= 10]
        
        x_pos = bathroom_groups.index
        ax2.bar(x_pos, bathroom_groups['mean'], alpha=0.7, color='coral', label='Mean')
        ax2.plot(x_pos, bathroom_groups['median'], 'ro-', label='Median', linewidth=2, markersize=8)
        ax2.set_title('Price vs Bathrooms', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Bathrooms')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'price_vs_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: price_vs_features.png")


def plot_correlation_heatmap(df, output_dir):
    """
    Plot correlation heatmap of numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    output_dir : Path
        Directory to save figures
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Warning: Not enough numeric columns for correlation heatmap")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")


def plot_geographic_scatter(df, output_dir):
    """
    Plot geographic scatterplot (lat, lon colored by price).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    output_dir : Path
        Directory to save figures
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("Warning: Latitude/longitude columns not found. Skipping geographic plot.")
        return
    
    # Remove rows with missing coordinates
    df_geo = df.dropna(subset=['latitude', 'longitude', 'price'])
    
    if len(df_geo) == 0:
        print("Warning: No valid geographic data. Skipping geographic plot.")
        return
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_geo['longitude'], df_geo['latitude'], 
                         c=df_geo['price'], cmap='viridis', 
                         alpha=0.6, s=10, edgecolors='none')
    plt.colorbar(scatter, label='Price ($)')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title('Geographic Distribution of Listings (Colored by Price)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'geographic_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: geographic_scatter.png")


def plot_price_by_room_type(df, output_dir):
    """
    Plot boxplot of price by room_type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe (may have room_type encoded as dummies)
    output_dir : Path
        Directory to save figures
    """
    # Check if we have room_type dummy variables
    room_type_cols = [col for col in df.columns if col.startswith('room_type_')]
    
    if not room_type_cols:
        print("Warning: Room type columns not found. Skipping room type plot.")
        return
    
    # Reconstruct room_type for visualization
    # Get the original room_type if it exists, or reconstruct from dummies
    if 'room_type' in df.columns:
        room_type_series = df['room_type']
    else:
        # Reconstruct from dummy variables
        room_type_series = pd.Series(index=df.index, dtype=str)
        for col in room_type_cols:
            room_type_name = col.replace('room_type_', '')
            room_type_series[df[col] == 1] = room_type_name
        
        # The first category (dropped in one-hot encoding) needs to be identified
        # We'll use a different approach: create a temporary column
        temp_df = df.copy()
        temp_df['room_type'] = 'Unknown'
        for col in room_type_cols:
            room_type_name = col.replace('room_type_', '')
            temp_df.loc[temp_df[col] == 1, 'room_type'] = room_type_name
        room_type_series = temp_df['room_type']
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    room_types = room_type_series.unique()
    data_to_plot = [df[room_type_series == rt]['price'].dropna() for rt in room_types if len(df[room_type_series == rt]) > 0]
    labels = [rt for rt in room_types if len(df[room_type_series == rt]) > 0]
    
    if data_to_plot:
        bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        plt.title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
        plt.xlabel('Room Type', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'price_by_room_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: price_by_room_type.png")
    else:
        print("Warning: No valid room type data for plotting.")


def run_eda(df):
    """
    Run all EDA visualizations and save them to output/figures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    """
    print("Running Exploratory Data Analysis...")
    print("=" * 50)
    
    output_dir = ensure_output_dir()
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    # Generate all plots
    plot_price_distribution(df, output_dir)
    plot_price_vs_features(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_geographic_scatter(df, output_dir)
    plot_price_by_room_type(df, output_dir)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print("=" * 50)
    print(f"Total records: {len(df)}")
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    
    print(f"\nFeature statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')
    print(df[numeric_cols].describe())
    
    print(f"\nAll figures saved to: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    # Test EDA
    from load_data import load_data
    from preprocess import preprocess
    
    df_raw = load_data()
    df = preprocess(df_raw)
    run_eda(df)
